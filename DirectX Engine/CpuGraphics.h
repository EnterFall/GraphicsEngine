#pragma once
#include "Vec2f.h"
#include "Vec3f.h"
#include "Matrix3.h"
#include "EFWin.h"
#include "MathHelper.h";
#include <memory>
#include <algorithm>
#include <cmath>
#include <numbers>

class CpuGraphics
{
public:
	const int bufferWidth = 1920;
	const int bufferHeight = 1080;
	const float fovLen = 800;

	const float fovScalar = 1.0f / (fovLen * fovLen);

	Vec3f cameraPos = Vec3f(0.0f, 0.0f, 0.0f);
	Vec3f cameraX90 = Vec3f(1.0f, 0.0f, 0.0f);
	Vec3f cameraY90 = Vec3f(0.0f, 1.0f, 0.0f);
	Vec3f cameraZ90 = Vec3f(0.0f, 0.0f, 1.0f);
	Vec3f cameraDirection = Vec3f(0.0f, 0.0f, 1.0f);
	Matrix3f transform;

	float travelSpeed = 4.0f;
	bool isMatrixTransform = true;
	
	
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
	void DrawProjectionTriangle(const Vec3f& v0, const Vec3f& v1, const Vec3f& v2, unsigned int color);
	void DrawRect(const Vec2f& v0, const Vec2f& v1, const Vec2f& v2, const Vec2f& v3, unsigned int color);
	void DrawCrosshair();

	Vec2f Transform(const Vec3f& camPos, const Vec3f& cam, const Vec3f& vertex) const;
	Vec2f TransformByMatrix(const Vec3f& vertex) const;
private:
	void DrawTriangleFlatBottom(const Vec2f& v0, const Vec2f& v1, const Vec2f& v2, unsigned int color);
	void DrawTriangleFlatTop(const Vec2f& v0, const Vec2f& v1, const Vec2f& v2, unsigned int color);
};

