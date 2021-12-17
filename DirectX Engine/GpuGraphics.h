#pragma once

//if enabled -> may break things
//#ifndef __CUDACC__  
//#define __CUDACC__
//#endif

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "CudaHelper.h"
#include "ZBuffer.h"
#include "Vec3.h"
#include "Color.h"
#include "Camera.h"
#include "Cube.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <algorithm>

class GpuGraphics
{
private:

public:
	int bufferWidth = 1920;
	int bufferHeight = 1080;
	float widthHalf = float(bufferWidth >> 1);
	float heightHalf = float(bufferHeight >> 1);
	float fov = MathHelper::PI_2;
	float travelSpeed = 4.0f;
	float zClip = 0.1f;
	Camera* camera;
	ZBuffer zBuffer;
	Color* screenBuffer;

	__device__ void Constructor(Color* sB, Camera* camCudaPtr);
	__device__ ~GpuGraphics();
	__device__ void DrawScreenTriangle(const Vec3f& v0, const Vec3f& v1, const Vec3f& v2, Color color);
	__device__ void Rasterize(float xLeft, float xRight, int index, float zLeft, float dz, Color color);
	__device__ void DrawTriangle(const Vec3f& v0, const Vec3f& v1, const Vec3f& v2, Color color);
	__device__ void DrawRect(const Vec3f& v0, const Vec3f& v1, const Vec3f& v2, const Vec3f& v3, Color color);
	__device__ void DrawPoligon(Vec3f* points, size_t count, Color color);
	__device__ void Draw(const Cube& cube, Color color);
	__device__ void DrawTriangleFromTo(const Vec3f& leftS, const Vec3f& leftE, const Vec3f& rightS, const Vec3f& rightE, Color color);
private:
	__device__ Vec3f Transform(const Vec3f& vertex) const;
	__device__ void Clip(Vec3f* list, int& listCount, const Vec3f& v0, const Vec3f& v1);
};

__global__ void ConstructGpuGraphics(GpuGraphics* gCudaPtr, Color* screenBufCudaPtr, Camera* camCudaPtr);
__global__ void DrawCubes(GpuGraphics* gCudaPtr, Cube* cubes, int count, int length);
__global__ void DrawTriangleFromTo(GpuGraphics* gCudaPtr, Vec3f* tr, int trCount, Color color);
__global__ void Clear(GpuGraphics* gCudaPtr);
__global__ void RasterizeGlobal(GpuGraphics* gCudaPtr, int yStart, int yEnd, float xLeft, float xRight, float zLeft,
	float dxLeft, float dxRight, float dzLeft, float dz, int bufferWidth, Color color);