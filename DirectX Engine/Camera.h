#pragma once
#include "Vec3f.h"
#include "Matrix3.h"
#include "MathHelper.h"

class Camera
{
public:
	Vec3f pos = Vec3f();
	Vec3f X90 = Vec3f(1.0f, 0.0f, 0.0f);
	Vec3f Y90 = Vec3f(0.0f, 1.0f, 0.0f);
	Vec3f Z90 = Vec3f(0.0f, 0.0f, 1.0f);

	Matrix3f transform;

	float fovW;
	float fovH;
	float scaleX;
	float scaleY;

	Vec3f leftNormal;
	Vec3f rightNormal;
	Vec3f topNormal;
	Vec3f bottomNormal;

	Camera(int width, int height, float fov);

	void Rotate(float x, float y);

};

