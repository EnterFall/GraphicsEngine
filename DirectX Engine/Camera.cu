#include "Camera.h"

__host__ __device__ Camera::Camera(int width, int height, float fov)
{
	pos = Vec3f();
	X90 = Vec3f(1.0f, 0.0f, 0.0f);
	Y90 = Vec3f(0.0f, 1.0f, 0.0f);
	Z90 = Vec3f(0.0f, 0.0f, 1.0f);

	widthHalf = width >> 1;
	heightHalf = height >> 1;
	fovW = fov;
	scale = widthHalf / tan(fovW / 2.0f);
	fovH = atan(heightHalf / scale) * 2;

	float rotFovW = MathHelper::PI_2 - fovW / 2.0f;
	float rotFovH = MathHelper::PI_2 - fovH / 2.0f;

	leftNormal = Vec3f(sin(rotFovW), 0.0f, cos(rotFovW));
	rightNormal = Vec3f(-leftNormal.x, 0.0f, leftNormal.z);

	bottomNormal = Vec3f(0.0f, sin(rotFovH), cos(rotFovH));
	topNormal = Vec3f(0.0f, -bottomNormal.y, bottomNormal.z);
}

__host__ __device__ void Camera::Rotate(float x, float y)
{
	Z90 = Vec3f(cos(y) * sin(x), sin(y), cos(y) * cos(x));
	Y90 = Vec3f(cos(y + MathHelper::PI_2) * sin(x), sin(y + MathHelper::PI_2), cos(y + MathHelper::PI_2) * cos(x));
	X90 = Y90.Cross(Z90);

	transform = Matrix3f(X90, Y90, Z90).Inverse2();
}

__host__ __device__ Vec3f Camera::ToScreen(const Vec3f& v) const
{
	auto projScale = scale / v.z;
	return Vec3f(widthHalf + v.x * projScale, heightHalf - v.y * projScale, 1.0f / v.z);
}