#include "Camera.h"

Camera::Camera(int width, int height, float fov)
{
	fovW = fov;
	scaleX = (width >> 1) / tan(fovW / 2.0f);
	fovH = atan((height >> 1) / scaleX) * 2;
	scaleY = (height >> 1) / tan(fovH / 2.0f);

	float rotFovW = MathHelper::PI_2_f - fovW / 2.0f;
	float rotFovH = MathHelper::PI_2_f - fovH / 2.0f;

	leftNormal = Vec3f(sin(rotFovW), 0.0f, cos(rotFovW));
	rightNormal = Vec3f(-leftNormal.x, 0.0f, leftNormal.z);

	bottomNormal = Vec3f(0.0f, sin(rotFovH), cos(rotFovH));
	topNormal = Vec3f(0.0f, -bottomNormal.y, bottomNormal.z);
}

void Camera::Rotate(float x, float y)
{
	Z90 = Vec3f(cos(y) * sin(x), sin(y), cos(y) * cos(x));
	Y90 = Vec3f(cos(y + MathHelper::PI_2_f) * sin(x), sin(y + MathHelper::PI_2_f), cos(y + MathHelper::PI_2_f) * cos(x));
	X90 = Y90.Cross(Z90);

	transform = Matrix3f(X90, Y90, Z90).Inverse2();
}
