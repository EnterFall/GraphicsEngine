#pragma once
#include <memory>
#include <algorithm>
#include <cmath>
#include <numbers>
#include <vector>

#include "Vec2f.h"
#include "Vec3.h"
#include "Matrix3.h"
#include "EFWin.h"
#include "MathHelper.h"
#include "Color.h"
#include "Camera.h"
#include "ZBuffer.h"
#include "Cube.h"

class CpuGraphics
{
public:
	const int bufferWidth = 1920;
	const int bufferHeight = 1080;
	const float widthHalf = bufferWidth >> 1;
	const float heightHalf = bufferHeight >> 1;
	double fov = MathHelper::PI_2;
	double travelSpeed = 4.0;
	bool isMatrixTransform = true;
	Camera camera = Camera(bufferWidth, bufferHeight, fov);
	ZBuffer zBuffer = ZBuffer(bufferWidth * bufferHeight);
	double zClip = 0.1;
private:
	std::shared_ptr<Color[]> screenBuffer;
	std::vector<Vec3d> clipBuffer;
	BITMAPINFO bufferInfo;
public:
	CpuGraphics();
	Color* GetScreenBuffer() const;
	const BITMAPINFO& GetBufferInfo() const;
	void Clear();
	// Use SetPixel to draw is very slow
	void SetPixel(int x, int y, Color color);
	void DrawRect(int x0, int y0, int x1, int y1, Color color);
	void DrawScreenTriangle(const Vec3d& v0, const Vec3d& v1, const Vec3d& v2, Color color);
	void DrawTriangle(const Vec3d& v0, const Vec3d& v1, const Vec3d& v2, Color color);
	void DrawRect(const Vec3d& v0, const Vec3d& v1, const Vec3d& v2, const Vec3d& v3, Color color);
	void DrawCube(const Vec3d& p0, const Vec3d& p1, Color color);
	void DrawPoligon(Vec3d* points, size_t count, Color color);
	void DrawCrosshair();
	void Draw(const Cube& cube, Color color);
private:
	Vec3d Transform(const Vec3d& vertex) const;
	Vec3d TransformByMatrix(const Vec3d& vertex) const;
	void DrawTriangleFromTo(const Vec3d& leftS, const Vec3d& leftE, const Vec3d& rightS, const Vec3d& rightE, Color color);
	void Clip(std::vector<Vec3d>* list, const Vec3d& v0, const Vec3d& v1);
	void ClipFull(std::vector<Vec3d>* list, const Vec3d& v0, const Vec3d& v1);
};

