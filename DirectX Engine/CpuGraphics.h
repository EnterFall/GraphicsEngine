#pragma once
#include "Vec2f.h"
#include "Vec3f.h"
#include "Matrix3.h"
#include "EFWin.h"
#include "MathHelper.h"
#include "Color.h"
#include "Camera.h"
#include "ZBuffer.h"
#include <memory>
#include <algorithm>
#include <cmath>
#include <numbers>
#include <vector>

class CpuGraphics
{
public:
	const int bufferWidth = 1920;
	const int bufferHeight = 1080;
	const float widthHalf = bufferWidth >> 1;
	const float heightHalf = bufferHeight >> 1;
	float fov = MathHelper::PI_2_f;
	float travelSpeed = 4.0f;
	bool isMatrixTransform = true;
	Camera camera = Camera(bufferWidth, bufferHeight, fov);
	ZBuffer zBuffer = ZBuffer(bufferWidth * bufferHeight);
private:
	std::shared_ptr<Color[]> screenBuffer;
	std::vector<Vec3f> clipBuffer;
	BITMAPINFO bufferInfo;
public:
	CpuGraphics();
	Color* GetScreenBuffer() const;
	const BITMAPINFO& GetBufferInfo() const;
	void Clear();
	// Use SetPixel to draw is very slow
	void SetPixel(int x, int y, Color color);
	void DrawRect(int x0, int y0, int x1, int y1, Color color);
	void DrawScreenTriangle(const Vec3f& v0, const Vec3f& v1, const Vec3f& v2, Color color);
	void DrawTriangle(const Vec3f& v0, const Vec3f& v1, const Vec3f& v2, Color color);
	void DrawRect(const Vec3f& v0, const Vec3f& v1, const Vec3f& v2, const Vec3f& v3, Color color);
	void DrawCube(const Vec3f& p0, const Vec3f& p1, Color color);
	void DrawPoligon(Vec3f* points, size_t count, Color color);
	void DrawCrosshair();
private:
	Vec3f Transform(const Vec3f& vertex) const;
	Vec3f TransformByMatrix(const Vec3f& vertex) const;
	void DrawTriangleFromTo(const Vec3f& leftS, const Vec3f& leftE, const Vec3f& rightS, const Vec3f& rightE, Color color);
	void Clip(std::vector<Vec3f>* list, const Vec3f& v0, const Vec3f& v1);
};

