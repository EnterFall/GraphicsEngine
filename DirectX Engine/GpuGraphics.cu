#include "GpuGraphics.h"

template <class T>
__device__ void swap(T& a, T& b)
{
	T temp = a;
	a = b;
	b = temp;
}

template <class T>
__device__ T min(T a, T b) {
	return a < b ? a : b;
}

template <class T>
__device__ T max(T a, T b) {
	return a > b ? a : b;
}

template <class T>
__device__ T clamp(T n, T lower, T upper) {
	return max(lower, min(n, upper));
}

__global__ void ConstructGpuGraphics(GpuGraphics* gCudaPtr, Color* screenBufCudaPtr, Camera* camCudaPtr)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id == 0)
	{
		gCudaPtr->Constructor(screenBufCudaPtr, camCudaPtr);
	}
}

__global__ void DrawCubes(GpuGraphics* gCudaPtr, Cube* cubes, int count, int length)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id < count)
	{
		auto vvv = cubes[id].verts[0];

		double scale = (256.0 / double(length));
		double xRaw = vvv.x * scale;
		double yRaw = vvv.y * scale;
		double zRaw = vvv.z * scale;
		unsigned char x = (unsigned char)(int)xRaw;
		unsigned char y = (unsigned char)(int)yRaw;
		unsigned char z = (unsigned char)(int)zRaw;
		gCudaPtr->Draw(cubes[id], Color(x, y, z));
	}
}

__global__ void Clear(GpuGraphics* gCudaPtr)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	gCudaPtr->screenBuffer[j * gCudaPtr->bufferWidth + i] = 0;
	gCudaPtr->zBuffer.buffer[j * gCudaPtr->bufferWidth + i] = 9.9e300;
}

__device__ void GpuGraphics::Constructor(Color* screenBufCudaPtr, Camera* camCudaPtr)
{
	bufferWidth = 1920;
	bufferHeight = 1080;
	widthHalf = bufferWidth >> 1;
	heightHalf = bufferHeight >> 1;
	fov = MathHelper::PI_2;
	travelSpeed = 4.0;
	zClip = 0.1;

	//camera = new Camera(bufferWidth, bufferHeight, fov);
	zBuffer = ZBuffer(bufferWidth * bufferHeight);
	screenBuffer = screenBufCudaPtr;
	camera = camCudaPtr;
}

__device__ GpuGraphics::~GpuGraphics()
{
	delete camera;
	delete[] screenBuffer;
}

__device__ void GpuGraphics::DrawScreenTriangle(const Vec3d& v0, const Vec3d& v1, const Vec3d& v2, Color color)
{
	const Vec3d* p0 = &v0;
	const Vec3d* p1 = &v1;
	const Vec3d* p2 = &v2;

	if (p0->y > p1->y) swap(p0, p1);
	if (p1->y > p2->y) swap(p1, p2);
	if (p0->y > p1->y) swap(p0, p1);

	if (p0->y != p1->y && p1->y != p2->y)
	{
		auto r = *p2 - *p0;
		Vec3d cross = Vec3d(p0->x + r.x * ((p1->y - p0->y) / r.y), p1->y, p0->z + r.z * ((p1->y - p0->y) / r.y));
		const Vec3d* c = &cross;

		if (c->x < p1->x) swap(c, p1);
		DrawTriangleFromTo(*p0, *p1, *p0, *c, color);
		DrawTriangleFromTo(*p1, *p2, *c, *p2, color);
	}
	else if (p1->y == p2->y)
	{
		if (p1->x > p2->x) swap(p1, p2);
		DrawTriangleFromTo(*p0, *p1, *p0, *p2, color);
	}
	else if (p0->y == p1->y)
	{
		if (p0->x > p1->x) swap(p0, p1);
		DrawTriangleFromTo(*p0, *p2, *p1, *p2, color);
	}
}

__device__ void GpuGraphics::DrawTriangleFromTo(const Vec3d& leftS, const Vec3d& leftE, const Vec3d& rightS, const Vec3d& rightE, Color color)
{
	Vec3d vLeft = leftE - leftS;
	Vec3d vRight = rightE - rightS;

	double dxLeft = vLeft.x / vLeft.y;
	double dxRight = vRight.x / vRight.y;

	double dzLeft = vLeft.z / vLeft.y;

	double outLeft = clamp(-leftS.y / vLeft.y, 0.0, 1.0);
	double outRight = clamp(-rightS.y / vRight.y, 0.0, 1.0);

	int yStart = clamp((int)ceil(leftS.y - 0.5), 0, bufferHeight);
	int yEnd = clamp((int)ceil(rightE.y - 0.5), 0, bufferHeight);

	// if y < 0, start vectors will be corrected
	Vec3d leftCorr = leftS + vLeft * outLeft;
	Vec3d rightCorr = rightS + vRight * outRight;
	
	double xLeft = leftCorr.x + dxLeft * fmod(0.5 - leftCorr.y + double(yStart), 1.0);
	double xRight = rightCorr.x + dxRight * fmod(0.5 - rightCorr.y + double(yStart), 1.0);

	double zLeft = leftCorr.z + dzLeft * fmod(0.5 - leftCorr.y + double(yStart), 1.0);

	Vec3d d = ((rightE + rightS) * 0.5) - ((leftE + leftS) * 0.5);
	double dz = d.z / d.x;
	int index = yStart * bufferWidth;
	for (int y = yStart; y < yEnd; y++)
	{
		int xStart = clamp((int)ceil(xLeft - 0.5), 0, bufferWidth);
		int xEnd = clamp((int)ceil(xRight - 0.5), 0, bufferWidth);
		double z = zLeft + dz * (clamp(xLeft, 0.0, (double)bufferWidth) - xLeft);
		for (int x = xStart; x < xEnd; x++)
		{
			if (zBuffer.Update(index + x, 1.0 / z))
			{
				screenBuffer[index + x] = color;
			}
			z += dz;
		}
		xLeft += dxLeft;
		xRight += dxRight;
		zLeft += dzLeft;
		index += bufferWidth;
	}
}

__device__ void GpuGraphics::DrawTriangle(const Vec3d& v0, const Vec3d& v1, const Vec3d& v2, Color color)
{
	// Backface culling, triangle in clockwise order is visible
	if ((v1 - v0).Cross(v2 - v0).Dot(v0 - camera->pos) < 0.0)
	{
		Vec3d val0 = Transform(v0);
		Vec3d val1 = Transform(v1);
		Vec3d val2 = Transform(v2);

		if (val0.z < zClip && val1.z < zClip && val2.z < zClip)
			return;

		// if vertices not clipped, clipBuffer may contains same elements
		if (val0.z < zClip || val1.z < zClip || val2.z < zClip)
		{
			Vec3d cBuf[6];
			int cBufIndex = 0;
			Clip(cBuf, cBufIndex, val0, val1);
			Clip(cBuf, cBufIndex, val1, val2);
			Clip(cBuf, cBufIndex, val2, val0);

			DrawPoligon(cBuf, cBufIndex, color);
		}
		else
		{
			DrawScreenTriangle(camera->ToScreen(val0), camera->ToScreen(val1), camera->ToScreen(val2), color);
		}
	}
}

// Only Z clipping
__device__ void GpuGraphics::Clip(Vec3d* list, int& index, const Vec3d& v0, const Vec3d& v1)
{
	const Vec3d* p0 = &v0;
	const Vec3d* p1 = &v1;

	if (p0->z < zClip && p1->z > zClip)
	{
		double scale = (zClip - p0->z) / (p1->z - p0->z);
		auto corr = *p0 + ((*p1 - *p0) * scale);

		list[index++] = camera->ToScreen(corr);
		list[index++] = camera->ToScreen(*p1);
	}
	else if (p0->z > zClip && p1->z < zClip)
	{
		double scale = (zClip - p1->z) / (p0->z - p1->z);
		auto corr = *p1 + ((*p0 - *p1) * scale);

		list[index++] = camera->ToScreen(*p0);
		list[index++] = camera->ToScreen(corr);
	}
	else if (p0->z > zClip && p1->z > zClip)
	{
		list[index++] = camera->ToScreen(*p0);
		list[index++] = camera->ToScreen(*p1);
	}
}

__device__ void GpuGraphics::DrawRect(const Vec3d& v0, const Vec3d& v1, const Vec3d& v2, const Vec3d& v3, Color color)
{
	DrawScreenTriangle(v0, v1, v2, color);
	DrawScreenTriangle(v0, v2, v3, color);
}

__device__ void GpuGraphics::DrawPoligon(Vec3d* points, size_t count, Color color)
{
	auto v0 = points;
	auto v1 = points + 1;
	auto v2 = points + 2;
	int i = 2;
	while (i < count)
	{
		if (*v1 != *v2)
		{
			DrawScreenTriangle(*v0, *v1, *v2, color);
		}
		v1 = v2;
		v2 = points + 1 + i;
		i++;
	}
}

__device__ void GpuGraphics::Draw(const Cube& cube, Color color)
{
	const Vec3d* v = cube.verts;
	//if (Vec3d(0.0, 0.0, -1.0).Dot(v[0] - camera.pos) < 0.0)
	{
		DrawTriangle(v[0], v[2], v[1], color);
		DrawTriangle(v[1], v[2], v[3], color);
	}
	//if (Vec3d(-1.0, 0.0, 0.0).Dot(v[0] - camera.pos) < 0.0)
	{
		DrawTriangle(v[0], v[4], v[2], color);
		DrawTriangle(v[4], v[6], v[2], color);
	}
	//if (Vec3d(1.0, 0.0, 0.0).Dot(v[1] - camera.pos) < 0.0)
	{
		DrawTriangle(v[1], v[3], v[5], color);
		DrawTriangle(v[5], v[3], v[7], color);
	}
	//if (Vec3d(0.0, -1.0, 0.0).Dot(v[0] - camera.pos) < 0.0)
	{
		DrawTriangle(v[0], v[1], v[4], color);
		DrawTriangle(v[4], v[1], v[5], color);
	}
	//if (Vec3d(0.0, 1.0, 0.0).Dot(v[2] - camera.pos) < 0.0)
	{
		DrawTriangle(v[2], v[6], v[7], color);
		DrawTriangle(v[7], v[3], v[2], color);
	}
	//if (Vec3d(0.0, 0.0, 1.0).Dot(v[4] - camera.pos) < 0.0)
	{
		DrawTriangle(v[4], v[5], v[7], color);
		DrawTriangle(v[7], v[6], v[4], color);
	}
}

__device__ Vec3d GpuGraphics::Transform(const Vec3d& vertex) const
{
	Vec3d camToVert = vertex - camera->pos;

	auto projX = camera->X90.Dot(camToVert);
	auto projY = camera->Y90.Dot(camToVert);
	auto projZ = camera->Z90.Dot(camToVert);

	return Vec3d(projX, projY, projZ);
}