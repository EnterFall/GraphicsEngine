#include "GpuGraphics.h"

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
	int idFull = blockIdx.x * blockDim.x + threadIdx.x;
	int idReal = idFull / 12;

	if (idReal < count)
	{
		auto vvv = cubes[idReal].verts[0];

		float scale = (256.0f / float(length));
		float xRaw = vvv.x * scale;
		float yRaw = vvv.y * scale;
		float zRaw = vvv.z * scale;
		unsigned char x = (unsigned char)(int)xRaw;
		unsigned char y = (unsigned char)(int)yRaw;
		unsigned char z = (unsigned char)(int)zRaw;

		int id = idFull % 12;
		const Vec3f* v = cubes[idReal].verts;
		int3 ind;

		if (id == 0)
			ind = int3{ 0, 2, 1 };
		if (id == 1)
			ind = int3{ 1, 2, 3 };
		if (id == 2)
			ind = int3{ 0, 4, 2 };
		if (id == 3)
			ind = int3{ 4, 6, 2 };
		if (id == 4)
			ind = int3{ 1, 3, 5 };
		if (id == 5)
			ind = int3{ 5, 3, 7 };
		if (id == 6)
			ind = int3{ 0, 1, 4 };
		if (id == 7)
			ind = int3{ 4, 1, 5 };
		if (id == 8)
			ind = int3{ 2, 6, 7 };
		if (id == 9)
			ind = int3{ 7, 3, 2 };
		if (id == 10)
			ind = int3{ 4, 5, 7 };
		if (id == 11)
			ind = int3{ 7, 6, 4 };

		gCudaPtr->DrawTriangle(v[ind.x], v[ind.y], v[ind.z], Color(x, y, z));
	}
}

__global__ void DrawTriangleFromTo(GpuGraphics* gCudaPtr, Vec3f* tr, int trCount, Color color)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < trCount)
	{
		int index = i * 4;
		gCudaPtr->DrawTriangleFromTo(tr[index], tr[index + 1], tr[index + 2], tr[index + 3], color);
	}
}

__global__ void Clear(GpuGraphics* gCudaPtr)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	gCudaPtr->screenBuffer[j * gCudaPtr->bufferWidth + i] = 0;
	gCudaPtr->zBuffer.buffer[j * gCudaPtr->bufferWidth + i] = 9.9e300;
}

__global__ void RasterizeGlobal(GpuGraphics* gCudaPtr, int yStart, int yEnd, float xLeft, float xRight, float zLeft, 
	float dxLeft, float dxRight, float dzLeft, float dz, int bufferWidth, Color color)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int yCount = yEnd - yStart;
	if (i < yCount)
	{
		xLeft = xLeft + i * dxLeft;
		xRight = xRight + i * dxRight;
		zLeft = zLeft + i * dzLeft;
		int index = (i + yStart) * bufferWidth;
		gCudaPtr->Rasterize(xLeft, xRight, index, zLeft, dz, color);
	}
}

__device__ void GpuGraphics::Constructor(Color* screenBufCudaPtr, Camera* camCudaPtr)
{
	bufferWidth = 1920;
	bufferHeight = 1080;
	widthHalf = bufferWidth >> 1;
	heightHalf = bufferHeight >> 1;
	fov = MathHelper::PI_2;
	travelSpeed = 4.0f;
	zClip = 0.1f;

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

__device__ void GpuGraphics::DrawScreenTriangle(const Vec3f& v0, const Vec3f& v1, const Vec3f& v2, Color color)
{
	const Vec3f* p0 = &v0;
	const Vec3f* p1 = &v1;
	const Vec3f* p2 = &v2;

	if (p0->y > p1->y) swap(p0, p1);
	if (p1->y > p2->y) swap(p1, p2);
	if (p0->y > p1->y) swap(p0, p1);

	if (p0->y != p1->y && p1->y != p2->y)
	{
		auto r = *p2 - *p0;
		Vec3f cross = Vec3f(p0->x + r.x * ((p1->y - p0->y) / r.y), p1->y, p0->z + r.z * ((p1->y - p0->y) / r.y));
		const Vec3f* c = &cross;

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

__device__ void GpuGraphics::Rasterize(float xLeft, float xRight, int index, float zLeft, float dz, Color color)
{
	int xStart = clamp((int)ceil(xLeft - 0.5f), 0, bufferWidth);
	int xEnd = clamp((int)ceil(xRight - 0.5f), 0, bufferWidth);
	float z = zLeft + dz * (clamp(xLeft, 0.0f, (float)bufferWidth) - xLeft);
	for (int x = xStart; x < xEnd; x++)
	{
		if (zBuffer.Update(index + x, 1.0f / z))
		{
			screenBuffer[index + x] = color;
		}
		z += dz;
	}
}

__device__ void GpuGraphics::DrawTriangleFromTo(const Vec3f& leftS, const Vec3f& leftE, const Vec3f& rightS, const Vec3f& rightE, Color color)
{
	Vec3f vLeft = leftE - leftS;
	Vec3f vRight = rightE - rightS;

	float dxLeft = vLeft.x / vLeft.y;
	float dxRight = vRight.x / vRight.y;

	float dzLeft = vLeft.z / vLeft.y;

	float outLeft = clamp(-leftS.y / vLeft.y, 0.0f, 1.0f);
	float outRight = clamp(-rightS.y / vRight.y, 0.0f, 1.0f);

	int yStart = clamp((int)ceil(leftS.y - 0.5f), 0, bufferHeight);
	int yEnd = clamp((int)ceil(rightE.y - 0.5f), 0, bufferHeight);

	// if y < 0, start vectors will be corrected
	Vec3f leftCorr = leftS + vLeft * outLeft;
	Vec3f rightCorr = rightS + vRight * outRight;
	
	float xLeft = leftCorr.x + dxLeft * fmod(0.5f - leftCorr.y + float(yStart), 1.0f);
	float xRight = rightCorr.x + dxRight * fmod(0.5f - rightCorr.y + float(yStart), 1.0f);

	float zLeft = leftCorr.z + dzLeft * fmod(0.5f - leftCorr.y + float(yStart), 1.0f);

	Vec3f d = ((rightE + rightS) * 0.5f) - ((leftE + leftS) * 0.5f);
	float dz = d.z / d.x;

	int index = yStart * bufferWidth;
	for (int y = yStart; y < yEnd; y++)
	{
		int xStart = clamp((int)ceil(xLeft - 0.5f), 0, bufferWidth);
		int xEnd = clamp((int)ceil(xRight - 0.5f), 0, bufferWidth);
		float z = zLeft + dz * (clamp(xLeft, 0.0f, (float)bufferWidth) - xLeft);
		for (int x = xStart; x < xEnd; x++)
		{
			if (zBuffer.Update(index + x, 1.0f / z))
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

__device__ void GpuGraphics::DrawTriangle(const Vec3f& v0, const Vec3f& v1, const Vec3f& v2, Color color)
{
	// Backface culling, triangle in clockwise order is visible
	if ((v1 - v0).Cross(v2 - v0).Dot(v0 - camera->pos) < 0.0f)
	{
		Vec3f val0 = Transform(v0);
		Vec3f val1 = Transform(v1);
		Vec3f val2 = Transform(v2);

		if (val0.z < zClip && val1.z < zClip && val2.z < zClip)
			return;

		// if vertices not clipped, clipBuffer may contains same elements
		if (val0.z < zClip || val1.z < zClip || val2.z < zClip)
		{
			Vec3f cBuf[6];
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
__device__ void GpuGraphics::Clip(Vec3f* list, int& index, const Vec3f& v0, const Vec3f& v1)
{
	const Vec3f* p0 = &v0;
	const Vec3f* p1 = &v1;

	if (p0->z < zClip && p1->z > zClip)
	{
		float scale = (zClip - p0->z) / (p1->z - p0->z);
		auto corr = *p0 + ((*p1 - *p0) * scale);

		list[index++] = camera->ToScreen(corr);
		list[index++] = camera->ToScreen(*p1);
	}
	else if (p0->z > zClip && p1->z < zClip)
	{
		float scale = (zClip - p1->z) / (p0->z - p1->z);
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

__device__ void GpuGraphics::DrawRect(const Vec3f& v0, const Vec3f& v1, const Vec3f& v2, const Vec3f& v3, Color color)
{
	DrawScreenTriangle(v0, v1, v2, color);
	DrawScreenTriangle(v0, v2, v3, color);
}

__device__ void GpuGraphics::DrawPoligon(Vec3f* points, size_t count, Color color)
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
	
}

__device__ Vec3f GpuGraphics::Transform(const Vec3f& vertex) const
{
	Vec3f camToVert = vertex - camera->pos;

	auto projX = camera->X90.Dot(camToVert);
	auto projY = camera->Y90.Dot(camToVert);
	auto projZ = camera->Z90.Dot(camToVert);

	return Vec3f(projX, projY, projZ);
}