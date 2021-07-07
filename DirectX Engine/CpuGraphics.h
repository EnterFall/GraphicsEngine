#pragma once
#include "Vec2f.h"
#include "Vec3f.h"
#include "Matrix3.h"
#include "EFWin.h"
#include "MathHelper.h"
#include "Camera.h"
#include <memory>
#include <algorithm>
#include <cmath>
#include <numbers>

class CpuGraphics
{
public:
	const int bufferWidth = 1920;
	const int bufferHeight = 1080;
	const float fov = MathHelper::PI_2_f;

	float travelSpeed = 4.0f;
	bool isMatrixTransform = true;
	
	Camera camera = Camera(bufferWidth, bufferHeight, fov);
	
private:
	std::shared_ptr<int[]> screenBuffer;
	BITMAPINFO bufferInfo;
public:
	CpuGraphics();

	int* GetBuffer() const;
	const BITMAPINFO& GetBufferInfo() const;

	void Clear();
	// Use SetPixel to draw is very slow
	void SetPixel(int x, int y, byte r, byte g, byte b);
	void DrawRect(int x0, int y0, int x1, int y1, unsigned int color);
	void DrawTriangle(const Vec2f& v0, const Vec2f& v1, const Vec2f& v2, unsigned int color);
	void DrawTriangle(const Vec3f& v0, const Vec3f& v1, const Vec3f& v2, unsigned int color);
	void DrawRect(const Vec2f& v0, const Vec2f& v1, const Vec2f& v2, const Vec2f& v3, unsigned int color);
	void DrawCube(const Vec3f& p0, const Vec3f& p1, unsigned int color);
	void DrawPoligon(const std::initializer_list<Vec2f>& points, unsigned int color);
	void DrawCrosshair();

	Vec2f Transform(const Vec3f& vertex) const;
	Vec2f TransformByMatrix(const Vec3f& vertex) const;
private:
	void DrawTriangleFlatBottom(const Vec2f& v0, const Vec2f& v1, const Vec2f& v2, unsigned int color);
	void DrawTriangleFlatTop(const Vec2f& v0, const Vec2f& v1, const Vec2f& v2, unsigned int color);
};

