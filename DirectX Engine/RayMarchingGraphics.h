#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "CudaHelper.h"
#include "Color.h"
#include "Camera.h"
#include <memory>

class RayMarchingGraphics
{
public:
	int bufferWidth;
	int bufferHeight;
	int widthHalf;
	int heightHalf;
	float minRayDistance;
	cudaSurfaceObject_t screenBuffer;

	__host__ static RayMarchingGraphics* Create(int width, int height, cudaSurfaceObject_t screenBuffer);
	__host__ void Draw(Camera* cameraCudaPtr, int width, int height, float sphereRadius);

	__host__ __device__ RayMarchingGraphics() = default;
	__device__ float DistanceToSphere(const Vec3f& dir, float sphereRadius);
	__device__ float DistanceToCube(const Vec3f& dir, float cubeRadius);
	__device__ float DistanceToObj(const Vec3f& dir, float radius);
	__device__ void DrawDevice(Camera* camera_DevPtr, float sphereRadius);
	__device__ void Fractal(Camera* camera_DevPtr, float zoom);
private:
	__host__ void Init(int width, int height, cudaSurfaceObject_t screenBuffer);
	__device__ float IntegrateLight(float distanceS, float distanceE, float length);
	__device__ float DEMandelbulb(Vec3f ray_pos);
};