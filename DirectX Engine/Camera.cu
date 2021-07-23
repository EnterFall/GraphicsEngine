#include "Camera.h"

__host__ __device__ Camera::Camera(int width, int height, double fov)
{
	pos = Vec3d();
	X90 = Vec3d(1.0, 0.0, 0.0);
	Y90 = Vec3d(0.0, 1.0, 0.0);
	Z90 = Vec3d(0.0, 0.0, 1.0);

	widthHalf = width >> 1;
	heightHalf = height >> 1;
	fovW = fov;
	scaleX = widthHalf / tan(fovW / 2.0);
	fovH = atan(heightHalf / scaleX) * 2;
	scaleY = heightHalf / tan(fovH / 2.0);

	double rotFovW = MathHelper::PI_2 - fovW / 2.0;
	double rotFovH = MathHelper::PI_2 - fovH / 2.0;

	leftNormal = Vec3d(sin(rotFovW), 0.0, cos(rotFovW));
	rightNormal = Vec3d(-leftNormal.x, 0.0, leftNormal.z);

	bottomNormal = Vec3d(0.0, sin(rotFovH), cos(rotFovH));
	topNormal = Vec3d(0.0, -bottomNormal.y, bottomNormal.z);
}

__host__ __device__ void Camera::Rotate(double x, double y)
{
	Z90 = Vec3d(cos(y) * sin(x), sin(y), cos(y) * cos(x));
	Y90 = Vec3d(cos(y + MathHelper::PI_2) * sin(x), sin(y + MathHelper::PI_2), cos(y + MathHelper::PI_2) * cos(x));
	X90 = Y90.Cross(Z90);

	transform = Matrix3f(X90, Y90, Z90).Inverse2();
}

__host__ __device__ Vec3d Camera::ToScreen(const Vec3d& v) const
{
	auto projScaleX = scaleX / v.z;
	auto projScaleY = scaleY / v.z;
	return Vec3d(widthHalf + v.x * projScaleX, heightHalf - v.y * projScaleY, 1.0 / v.z);
}