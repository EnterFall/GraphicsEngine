#pragma once
#include "cuda_runtime.h"
#include "Vec3.h"
#include "Matrix3.h"
#include "MathHelper.h"

class Camera
{
public:
	Vec3d pos;
	Vec3d X90;
	Vec3d Y90;
	Vec3d Z90;
	Matrix3f transform;
	Vec3d leftNormal;
	Vec3d rightNormal;
	Vec3d topNormal;
	Vec3d bottomNormal;
	double widthHalf;
	double heightHalf;
	double fovW;
	double fovH;
	double scaleX;
	double scaleY;
public:
	__host__ __device__ Camera(int width, int height, double fov);
	__host__ __device__ void Rotate(double x, double y);
	__host__ __device__ Vec3d ToScreen(const Vec3d& v) const;
};

