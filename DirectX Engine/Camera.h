#pragma once
#include "cuda_runtime.h"
#include "Vec3.h"
#include "Matrix3.h"
#include "MathHelper.h"

class Camera
{
public:
	Vec3f pos;
	Vec3f X90;
	Vec3f Y90;
	Vec3f Z90;
	Matrix3f transform;
	Vec3f leftNormal;
	Vec3f rightNormal;
	Vec3f topNormal;
	Vec3f bottomNormal;
	float widthHalf;
	float heightHalf;
	float fovW;
	float fovH;
	float scale;
public:
	__host__ __device__ Camera() = default;
	__host__ __device__ Camera(int width, int height, float fov);
	__host__ __device__ void Rotate(float x, float y);
	__host__ __device__ Vec3f ToScreen(const Vec3f& v) const;
};

