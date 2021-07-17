#pragma once
#include "Vec3.h"
#include "Matrix3.h"
#include "MathHelper.h"
#include "Vec2f.h"

class Camera
{
public:
	Vec3d pos = Vec3d();
	Vec3d X90 = Vec3d(1.0, 0.0, 0.0);
	Vec3d Y90 = Vec3d(0.0, 1.0, 0.0);
	Vec3d Z90 = Vec3d(0.0, 0.0, 1.0);
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
	Camera(int width, int height, double fov);
	void Rotate(double x, double y);
	Vec3d ToScreen(const Vec3d& v) const;
};

