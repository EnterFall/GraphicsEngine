#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "ZBuffer.h"
#include "Vec3.h"
#include "Color.h"
#include "Camera.h"
#include "Cube.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

class GpuGraphics
{
private:

public:
	int bufferWidth = 1920;
	int bufferHeight = 1080;
	double widthHalf = double(bufferWidth >> 1);
	double heightHalf = double(bufferHeight >> 1);
	double fov = MathHelper::PI_2;
	double travelSpeed = 4.0;
	double zClip = 0.1;
	Camera* camera;
	ZBuffer zBuffer;
	Color* screenBuffer;

	//__global__ static void ConstructGpuGraphics(GpuGraphics* gCudaPtr);
	//__global__ static void SetCamera(GpuGraphics* gCudaPtr, Camera* camCudaPtr);
	//__global__ static void DrawCubes(GpuGraphics* gCudaPtr, Cube* cubes, int count);
	//__global__ static void Clear(GpuGraphics* gCudaPtr);

	__device__ void Constructor(Color* sB, Camera* camCudaPtr);
	__device__ ~GpuGraphics();
	__device__ void DrawScreenTriangle(const Vec3d& v0, const Vec3d& v1, const Vec3d& v2, Color color);
	__device__ void DrawTriangle(const Vec3d& v0, const Vec3d& v1, const Vec3d& v2, Color color);
	__device__ void DrawRect(const Vec3d& v0, const Vec3d& v1, const Vec3d& v2, const Vec3d& v3, Color color);
	__device__ void DrawPoligon(Vec3d* points, size_t count, Color color);
	__device__ void Draw(const Cube& cube, Color color);
private:
	__device__ Vec3d Transform(const Vec3d& vertex) const;
	__device__ void DrawTriangleFromTo(const Vec3d& leftS, const Vec3d& leftE, const Vec3d& rightS, const Vec3d& rightE, Color color);
	__device__ void Clip(Vec3d* list, int& listCount, const Vec3d& v0, const Vec3d& v1);
};

template <class T>
__device__ void swap(T& a, T& b);
template <class T>
__device__ T min(T a, T b);
template <class T>
__device__ T max(T a, T b);
template <class T>
__device__ T clamp(T n, T lower, T upper);

__global__ void ConstructGpuGraphics(GpuGraphics* gCudaPtr, Color* screenBufCudaPtr, Camera* camCudaPtr);
__global__ void DrawCubes(GpuGraphics* gCudaPtr, Cube* cubes, int count, int length);
__global__ void Clear(GpuGraphics* gCudaPtr);