#include "CpuGraphics.h"

CpuGraphics::CpuGraphics()
{
	screenBuffer = std::make_shared<Color[]>(bufferWidth * bufferHeight);
	clipBuffer = std::vector<Vec2f>(6);

	for (int y = 0; y < bufferHeight; y++)
	{
		for (int x = 0; x < bufferWidth; x++)
		{
			screenBuffer[y * bufferWidth + x] = Color(x, y, 0);
		}
	}

	bufferInfo = BITMAPINFO();
	bufferInfo.bmiHeader.biSize = sizeof(bufferInfo.bmiHeader);
	bufferInfo.bmiHeader.biWidth = bufferWidth;
	bufferInfo.bmiHeader.biHeight = -bufferHeight;
	bufferInfo.bmiHeader.biBitCount = 32;
	bufferInfo.bmiHeader.biPlanes = 1;
	bufferInfo.bmiHeader.biCompression = BI_RGB;
}

Color* CpuGraphics::GetScreenBuffer() const
{
	return screenBuffer.get();
}

const BITMAPINFO& CpuGraphics::GetBufferInfo() const
{
	return bufferInfo;
}

// Slow 
void CpuGraphics::Clear()
{
	for (int y = 0; y < bufferHeight; y++)
	{
		for (int x = 0; x < bufferWidth; x++)
		{
			screenBuffer[y * bufferWidth + x] = Colors::Black;
		}
	}
}


void CpuGraphics::SetPixel(int x, int y, Color color)
{
	screenBuffer[y, x] = color;
}

void CpuGraphics::DrawRect(int x0, int y0, int x1, int y1, Color color)
{
	for (int y = y0; y < y1; y++)
	{
		for (int x = x0; x < x1; x++)
		{
			screenBuffer[y * bufferWidth + x] = color;
		}
	}
}

void CpuGraphics::DrawTriangle(const Vec2f& v0, const Vec2f& v1, const Vec2f& v2, Color color)
{
	const Vec2f* p0 = &v0;
	const Vec2f* p1 = &v1;
	const Vec2f* p2 = &v2;
	if (p0->y > p1->y) std::swap(p0, p1);
	if (p1->y > p2->y) std::swap(p1, p2);
	if (p0->y > p1->y) std::swap(p0, p1);

	if (p1->y == p2->y)
	{
		if (p1->x > p2->x) std::swap(p1, p2);
		DrawTriangleFlatBottom(*p0, *p1, *p2, color);
	}
	else if (p0->y == p1->y)
	{
		if (p0->x > p1->x) std::swap(p0, p1);
		DrawTriangleFlatTop(*p0, *p1, *p2, color);
	}
	else
	{
		auto r = *p2 - *p0;
		Vec2f cross = { p0->x + r.x * ((p1->y - p0->y) / r.y), p1->y };
		const Vec2f* c = &cross;

		if (c->x < p1->x) std::swap(c, p1);
		DrawTriangleFlatBottom(*p0, *p1, *c, color);
		DrawTriangleFlatTop(*p1, *c, *p2, color);
	}
}

// Sometimes pixel holes appear (
void CpuGraphics::DrawTriangleFlatBottom(const Vec2f& v0, const Vec2f& v1, const Vec2f& v2, Color color)
{
	Vec2f vLeft = v1 - v0;
	Vec2f vRight = v2 - v0;

	float dxLeft = vLeft.x / vLeft.y;
	float dxRight = vRight.x / vRight.y;

	float outLeft = std::clamp(-v0.y / vLeft.y, 0.f, 1.f);
	float outRight = std::clamp(-v0.y / vRight.y, 0.f, 1.f);

	int yStart = std::clamp((int)round(v0.y), 0, bufferHeight);
	int yEnd = std::clamp((int)ceil(v2.y - 0.5f), 0, bufferHeight);

	// if y < 0, start vectors will be corrected
	auto leftCorr = v0 + vLeft * outLeft;
	auto rightCorr = v0 + vRight * outRight;

	float xLeft = leftCorr.x + dxLeft * (0.5f - leftCorr.y + float(yStart));
	float xRight = rightCorr.x + dxRight * (0.5f - rightCorr.y + float(yStart));
	

	int index = yStart * bufferWidth;

	for (int y = yStart; y < yEnd; y++)
	{
		int xStart = std::clamp((int)ceil(xLeft - 0.5f), 0, bufferWidth);
		int xEnd =   std::clamp((int)ceil(xRight - 0.5f), 0, bufferWidth);
		std::fill_n(screenBuffer.get() + index + xStart, xEnd - xStart, color);
		xLeft += dxLeft;
		xRight += dxRight;

		index += bufferWidth;
	}
}

// Sometimes pixel holes appear (
void CpuGraphics::DrawTriangleFlatTop(const Vec2f& v0, const Vec2f& v1, const Vec2f& v2, Color color)
{
	Vec2f vLeft = v2 - v0;
	Vec2f vRight = v2 - v1;

	float dxLeft = vLeft.x / vLeft.y;
	float dxRight = vRight.x / vRight.y;

	float outLeft = std::clamp(-v0.y / vLeft.y, 0.f, 1.f);
	float outRight = std::clamp(-v1.y / vRight.y, 0.f, 1.f);

	int yStart = std::clamp((int)ceil(v0.y - 0.5f), 0, bufferHeight);
	int yEnd = std::clamp((int)ceil(v2.y - 0.5f), 0, bufferHeight);

	// if y < 0, start vectors will be corrected
	auto leftCorr = v0 + vLeft * outLeft;
	auto rightCorr = v1 + vRight * outRight;

	float t;
	float xLeft = leftCorr.x + dxLeft * modf(0.5f - leftCorr.y + float(yStart), &t);
	float xRight = rightCorr.x + dxRight * modf(0.5f - rightCorr.y + float(yStart), &t);

	int index = yStart * bufferWidth;

	for (int y = yStart; y < yEnd; y++)
	{
		int xStart = std::clamp((int)ceil(xLeft - 0.5f), 0, bufferWidth);
		int xEnd =   std::clamp((int)ceil(xRight - 0.5f), 0, bufferWidth);
		std::fill_n(screenBuffer.get() + index + xStart, xEnd - xStart, color);
		xLeft += dxLeft;
		xRight += dxRight;

		index += bufferWidth;
	}
}

void CpuGraphics::DrawTriangle(const Vec3f& v0, const Vec3f& v1, const Vec3f& v2, Color color)
{
	// Backface culling, triangle in clockwise order is visible
	if ((v1 - v0).Cross(v2 - v0).Dot(v0 - camera.pos) < 0)
	{
		Vec3f val0;
		Vec3f val1;
		Vec3f val2;
		if (isMatrixTransform)
		{
			val0 = TransformByMatrix(v0);
			val1 = TransformByMatrix(v1);
			val2 = TransformByMatrix(v2);
		}
		else
		{
			val0 = Transform(v0);
			val1 = Transform(v1);
			val2 = Transform(v2);
		}

		// if vertices not clipped, clipBuffer may contains same elements
		Clip(&clipBuffer, val0, val1);
		Clip(&clipBuffer, val1, val2);
		Clip(&clipBuffer, val2, val0);

		DrawPoligon(clipBuffer.begin()._Ptr, clipBuffer.size(), color);
		clipBuffer.clear();
	}
}

// Only Z clipping
void CpuGraphics::Clip(std::vector<Vec2f>* list, const Vec3f& v0, const Vec3f& v1)
{
	float zClip = 1.0f;

	const Vec3f* p0 = &v0;
	const Vec3f* p1 = &v1;

	if (p0->z < zClip && p1->z > zClip)
	{
		float scale = (zClip - p0->z) / (p1->z - p0->z);
		auto corr = *p0 + ((*p1 - *p0) * scale);

		list->push_back(camera.ToScreen(corr));
		list->push_back(camera.ToScreen(*p1));
	}
	else if (p0->z > zClip && p1->z < zClip)
	{
		float scale = (zClip - p1->z) / (p0->z - p1->z);
		auto corr = *p1 + ((*p0 - *p1) * scale);

		list->push_back(camera.ToScreen(*p0));
		list->push_back(camera.ToScreen(corr));
	}
	else if (p0->z > zClip && p1->z > zClip)
	{
		list->push_back(camera.ToScreen(*p0));
		list->push_back(camera.ToScreen(*p1));
	}
}

void CpuGraphics::DrawRect(const Vec2f& v0, const Vec2f& v1, const Vec2f& v2, const Vec2f& v3, Color color)
{
	DrawTriangle(v0, v1, v2, color);
	DrawTriangle(v0, v2, v3, color);
}

void CpuGraphics::DrawCube(const Vec3f& p0, const Vec3f& p1, Color color)
{
	auto dif = p1 - p0;
	auto x = Vec3f(dif.x, 0.0f, 0.0f);
	auto y = Vec3f(0.0f, dif.y, 0.0f);
	auto z = Vec3f(0.0f, 0.0f, dif.z);
	// Not optimized
	DrawTriangle(p0, p0 + x + y, p0 + x, color);
	DrawTriangle(p0, p0 + y, p0 + x + y, color);

	DrawTriangle(p0, p0 + x, p0 + x + z, color);
	DrawTriangle(p0, p0 + x + z, p0 + z, color);

	DrawTriangle(p0, p0 + y + z, p0 + y, color);
	DrawTriangle(p0, p0 + z, p0 + y + z, color);

	DrawTriangle(p1, p1 - x, p1 - x - y, color);
	DrawTriangle(p1, p1 - x - y, p1 - y, color);

	DrawTriangle(p1, p1 - x - z, p1 - x, color);
	DrawTriangle(p1, p1 - z, p1 - x - z, color);

	DrawTriangle(p1, p1 - y, p1 - y - z, color);
	DrawTriangle(p1, p1 - y - z, p1 - z, color);
}

void CpuGraphics::DrawPoligon(Vec2f* points, size_t count, Color color)
{
	auto v0 = points;
	auto v1 = points + 1;
	auto v2 = points + 2;
	int i = 2;
	while (i < count)
	{
		if (*v1 != *v2)
		{
			DrawTriangle(*v0, *v1, *v2, color);
		}
		v1 = v2;
		v2 = points + 1 + i;
		i++;
	}
}

void CpuGraphics::DrawCrosshair()
{
	int offset = 10;
	for (int y = heightHalf - offset; y < heightHalf + offset; y++)
	{
		for (int x = widthHalf - offset; x < widthHalf + offset; x++)
		{
			screenBuffer[y * bufferWidth + x] = Colors::Blue;
		}
	}
}

Vec3f CpuGraphics::Transform(const Vec3f& vertex) const
{
	Vec3f camToVert = vertex - camera.pos;

	auto projX = camera.X90.Dot(camToVert);
	auto projY = camera.Y90.Dot(camToVert);
	auto projZ = camera.Z90.Dot(camToVert);

	return Vec3f(projX, projY, projZ);
}

Vec3f CpuGraphics::TransformByMatrix(const Vec3f& vertex) const
{
	Vec3f camToVert = vertex - camera.pos;
	return camera.transform.Mult(camToVert);
}