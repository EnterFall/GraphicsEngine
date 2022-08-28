#include "RayMarchingGraphics.h"
#include "CudaHelper.h"

__global__ void RayMarchingGraphics_Draw(RayMarchingGraphics* obj, Camera* camera_DevPtr, float sphereRadius)
{
	__shared__ RayMarchingGraphics rayMarchG_shared;
	__shared__ Camera cam_shared;
	if (threadIdx.x == 0 && threadIdx.y == 0)
	{
		rayMarchG_shared = *obj;
		cam_shared = *camera_DevPtr;
	}
	__syncthreads();
	rayMarchG_shared.DrawDevice(&cam_shared, sphereRadius);
	//rayMarchG_shared.Fractal(&cam_shared, sphereRadius);
	//obj->DrawDevice(camera_DevPtr, sphereRadius);
}

__host__ RayMarchingGraphics* RayMarchingGraphics::Create(int width, int height, cudaSurfaceObject_t screenBuffer)
{
	RayMarchingGraphics* obj = CudaHelper::CreateManagedCudaObj<RayMarchingGraphics>();
	obj->Init(width, height, screenBuffer);
	return obj;
}

__host__ void RayMarchingGraphics::Init(int width, int height, cudaSurfaceObject_t screenBuffer)
{
	bufferWidth = width;
	bufferHeight = height;
	widthHalf = width >> 1;
	heightHalf = height >> 1;
	//minRayDistance = 0.001f;
	  minRayDistance = 0.001f;
	this->screenBuffer = screenBuffer;
}

__host__ void RayMarchingGraphics::Draw(Camera* cameraCudaPtr, int width, int height, float sphereRadius)
{
	int tileSide = 8;
	dim3 thread = dim3(tileSide, tileSide);
	dim3 block = dim3((width / tileSide) + 1, (height / tileSide) + 1);
	RayMarchingGraphics_Draw << <block, thread >> > (this, cameraCudaPtr, sphereRadius);
	CudaAssert(cudaGetLastError());
}

__device__ float RayMarchingGraphics::DistanceToSphere(const Vec3f& dir, float sphereRadius)
{
	return dir.Length() - sphereRadius;
}

__device__ float RayMarchingGraphics::DistanceToCube(const Vec3f& dir, float cubeRadius)
{
	Vec3f d = Vec3f(fabsf(dir.x), fabsf(dir.y), fabsf(dir.z)) - cubeRadius;
	d.x = fmaxf(d.x, 0.0f);
	d.y = fmaxf(d.y, 0.0f);
	d.z = fmaxf(d.z, 0.0f);
	return d.Length();
}

__device__ float RayMarchingGraphics::DistanceToObj(const Vec3f& dir, float radius)
{
	return fmaxf(-(DistanceToSphere(dir, radius * 1.10f)), DistanceToCube(dir, radius));
}

__device__ void RayMarchingGraphics::DrawDevice(Camera* camera, float sphereRadius)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	Vec3f light = Vec3f(1, 1, 1);
	Vec3f lightColor = Vec3f(0.1f, 1.0f, 0.8f);
	light.Normalize();
	float maxLimit = 100.0f;
	if (i < bufferWidth && j < bufferHeight)
	{
		float widthScale = i - widthHalf;
		float heightScale = -j + heightHalf;
		Vec3f x = camera->X90 * widthScale;
		Vec3f y = camera->Y90 * heightScale;
		Vec3f z = camera->Z90 * camera->scale;
		Vec3f dir = x + y + z;
		dir.Normalize();

		Vec3f rayPos = camera->pos;

		Vec3f objPos;
		float stepColor = 0.0f;
		float sumDist = 0.0f;
		float oldDist = 0.0f;
		for (int step = 0; step < 1000; step++)
		{
			//objPos.x = clamp(roundf(rayPos.x), 0.0f, maxLimit);
			//objPos.y = clamp(roundf(rayPos.y), 0.0f, maxLimit);
			//objPos.z = clamp(roundf(rayPos.z), 0.0f, maxLimit);
			objPos.x = 0.0f;
			objPos.y = 0.0f;
			objPos.z = 0.0f;

			Vec3f dis = objPos - rayPos;
			//float distance = DistanceToSphere(dis, sphereRadius);
			
			float distance = DEMandelbulb(rayPos, oldDist);

			if (distance > 10000.0f || stepColor >= 255.0f)
			{
				float color = fminf(stepColor, 255.0f);
				//float color = 255.0f;
				surf2Dwrite<int>(Color(color, color, color).dword, screenBuffer, i * sizeof(int), j);
				return;
			}
			//stepColor += fminf(0.5f * powf(distance, -1.0f), 1.0f);
			//stepColor += IntegrateLight(distance, DistanceToSphere(objPos - (rayPos + (dir * (distance * 0.90f))), sphereRadius), distance) * 0.05f;
			//dir * (dis.Length() / dir.Dot(dis)) - dis).Length() < sphereRadius

			if (distance < minRayDistance && distance < fmaxf(minRayDistance * sumDist, 0.000001f))
			{
				Vec3f d = rayPos - objPos;
				d.Normalize();
				//float lightScale = (d.Dot(light) + 1) * 0.25f + 0.5f;
				//float lightScale = 1.0f;
				//lightColor = objPos * (1.0f / maxLimit) * lightScale + stepColor;
				//lightColor.x = fminf(lightColor.x, 1.0f);
				//lightColor.y = fminf(lightColor.y, 1.0f);
				//lightColor.z = fminf(lightColor.z, 1.0f);
				lightColor = Vec3f(1.0f, 1.0f, 1.0f);

				//color = Vec3f(1.0f, 1.0f, 1.0f);

				float lightScale = 1.0f;
				float a = 1.0f / (1.0f + step * 0.02f);
				lightScale += (0.0f + a) * 1.0f;
				float rLen = rayPos.Length();
				surf2Dwrite<int>(Color(lightScale * 255.0f, lightScale * 255.0f, lightScale * 255.0f).dword, screenBuffer, i * sizeof(int), j);
				return;
			}
			rayPos += (dir * (distance * 1.00f));
			sumDist += distance;
			oldDist = distance;
		}
		float color = fminf(stepColor, 255.0f);
		surf2Dwrite<int>(Color(color, color, color).dword, screenBuffer, i * sizeof(int), j);
	}
}

__device__ float RayMarchingGraphics::IntegrateLight(float distanceS, float distanceE, float length)
{
	return fminf(fabsf(((-1.0f / distanceE) - (-1.0f / distanceS)) / fabsf(distanceE-distanceS)) * length, 200.0f);
}

__device__ void RayMarchingGraphics::Fractal(Camera* camera_DevPtr, float zoom)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if (i < bufferWidth && j < bufferHeight)
	{
		Vec3f c = Vec3f((i - widthHalf) * zoom, (j - heightHalf) * zoom, 1.0f);
		Vec3f z = Vec3f(0.0f, 0.0f, 0.0f);
		int numTier = 0;
		int maxTier = 40;
		for (numTier = 0; numTier < maxTier; numTier++)
		{
			z = Vec3f(z.x * z.x - z.y * z.y - c.x, 2 * z.x * z.y + c.y, 1.0f);
			if (z.x * z.x + z.y * z.y > 4.0f)
			{
				break;
			}
		}
		int col = (255.0f * numTier) / maxTier;
		surf2Dwrite<int>(Color(col, col, col).dword, screenBuffer, i * sizeof(int), j);
	}
}

__device__ float RayMarchingGraphics::DEMandelbulb(Vec3f ray_pos, float oldDist) {
	float POWER = 8.0f;
	float TARGET_POS[3] = { 0.0f, 0.0f, 0.0f };

	Vec3f tmp_pos = ray_pos;
	Vec3f cart_pos;
	float dr = 1.0f;
	float r;
	float theta;
	float phi;
	float zr;
	for (int tmp_iter = 0; tmp_iter < 16; tmp_iter++)
	{
		r = tmp_pos.Length();
		if (r > 2.0f) { break; }
		// approximate the distance differential
		dr = POWER * powf(r, POWER - 1.0f) * dr + 1.0f;
		// calculate fractal surface
		// convert to polar coordinates
		theta = POWER * acosf(tmp_pos.z / r);
		phi = POWER * atan2f(tmp_pos.y, tmp_pos.x);
		zr = powf(r, POWER);
		// convert back to cartesian coordinated
		cart_pos.x = zr * sinf(theta) * cosf(phi);
		cart_pos.y = zr * sinf(theta) * sinf(phi);
		cart_pos.z = zr * cosf(theta);
		tmp_pos = ray_pos + cart_pos;
	}
	// distance estimator
	return 0.5f * logf(r) * r / dr;
}

__device__ float RayMarchingGraphics::DEMandelbulbDouble(Vec3f ray_pos, double oldDist) {
	double POWER = 8.0;
	double TARGET_POS[3] = { 0.0, 0.0, 0.0 };

	Vec3d ray_pos2;

	ray_pos2.x = ray_pos.x;
	ray_pos2.y = ray_pos.y;
	ray_pos2.z = ray_pos.z;
	Vec3d tmp_pos = ray_pos2;
	Vec3d cart_pos;
	double dr = 1.0;
	double r;
	double theta;
	double phi;
	double zr;
	for (int tmp_iter = 0; tmp_iter < 16; tmp_iter++)
	{
		r = tmp_pos.Length();
		if (r > 2.0) { break; }
		// approximate the distance differential
		dr = POWER * pow(r, POWER - 1.0) * dr + 1.0;
		// calculate fractal surface
		// convert to polar coordinates
		theta = POWER * acos(tmp_pos.z / r);
		phi = POWER * atan2(tmp_pos.y, tmp_pos.x);
		zr = pow(r, POWER);
		// convert back to cartesian coordinated
		cart_pos.x = zr * sin(theta) * cos(phi);
		cart_pos.y = zr * sin(theta) * sin(phi);
		cart_pos.z = zr * cos(theta);
		tmp_pos = ray_pos2 + cart_pos;
	}
	// distance estimator
	return 0.5 * log(r) * r / dr;
}