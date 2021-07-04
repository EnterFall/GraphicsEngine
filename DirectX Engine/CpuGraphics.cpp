#include "CpuGraphics.h"

CpuGraphics::CpuGraphics()
{
	screenBuffer = std::make_shared<int[]>(bufferWidth * bufferHeight);

	for (int y = 0; y < bufferHeight; y++)
	{
		for (int x = 0; x < bufferWidth; x++)
		{
			screenBuffer[y * bufferWidth + x] = RGB(x, y, 0);
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

int* CpuGraphics::GetBuffer() const
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
			screenBuffer[y * bufferWidth + x] = 0x000000;
		}
	}
}


void CpuGraphics::SetPixel(int x, int y, byte r, byte g, byte b)
{
	screenBuffer[y, x] = ((COLORREF)(((BYTE)(b) | ((WORD)((BYTE)(g)) << 8)) | (((DWORD)(BYTE)(r)) << 16)));
}

void CpuGraphics::DrawRect(int x0, int y0, int x1, int y1, unsigned int color)
{
	for (int y = y0; y < y1; y++)
	{
		for (int x = x0; x < x1; x++)
		{
			screenBuffer[y * bufferWidth + x] = color;
		}
	}
	//StretchDIBits(hdc, 0, 0, width, height, 0, 0, width, height,  )
}

void CpuGraphics::DrawTriangle(const Vec2f& v0, const Vec2f& v1, const Vec2f& v2, unsigned int color)
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

void CpuGraphics::DrawTriangleFlatBottom(const Vec2f& v0, const Vec2f& v1, const Vec2f& v2, unsigned int color)
{
	Vec2f vLeft = v1 - v0;
	Vec2f vRight = v2 - v0;

	float dxLeft = vLeft.x / vLeft.y;
	float dxRight = vRight.x / vRight.y;

	float outLeft = std::clamp(-v0.y / vLeft.y, 0.f, 1.f);
	float outRight = std::clamp(-v0.y / vRight.y, 0.f, 1.f);

	//Vec2f v0Left = { v0.x + vLeft.x * outLeft, v0.y + vLeft.y * outLeft };
	//Vec2f v0Right = { v0.x + vRight.x * outRight, v0.y + vRight.y * outRight };

	float t;
	// Might be miscalculation in (0.5f - modf(v0.y, &t)) if y < 0
	float xLeft = v0.x + vLeft.x * outLeft + dxLeft * modf(0.5f - v0.y, &t);
	float xRight = v0.x + vRight.x * outRight + dxRight * modf(0.5f - v0.y, &t);
	
	int yStart = std::clamp((int)ceil(v0.y - 0.5f), 0, bufferHeight);
	int yEnd =   std::clamp((int)ceil(v2.y - 0.5f), 0, bufferHeight);
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

void CpuGraphics::DrawTriangleFlatTop(const Vec2f& v0, const Vec2f& v1, const Vec2f& v2, unsigned int color)
{
	Vec2f vLeft = v2 - v0;
	Vec2f vRight = v2 - v1;

	float dxLeft = vLeft.x / vLeft.y;
	float dxRight = vRight.x / vRight.y;

	float outLeft = std::clamp(-v0.y / vLeft.y, 0.f, 1.f);
	float outRight = std::clamp(-v1.y / vRight.y, 0.f, 1.f);

	float t;
	// Might be miscalculation in (0.5f - modf(v0|v1.y, &t)) if y < 0
	float xLeft = v0.x + vLeft.x * outLeft + dxLeft * modf(0.5f + v0.y, &t);
	float xRight = v1.x + vRight.x * outRight + dxRight * modf(0.5f + v1.y, &t);

	int yStart = std::clamp((int)ceil(v0.y - 0.5f), 0, bufferHeight);
	int yEnd =   std::clamp((int)ceil(v2.y - 0.5f), 0, bufferHeight);
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

void CpuGraphics::DrawProjectionTriangle(const Vec3f& v0, const Vec3f& v1, const Vec3f& v2, unsigned int color)
{

	Vec2f val0;
	Vec2f val1;
	Vec2f val2;
	if (isMatrixTransform)
	{
		val0 = TransformByMatrix(v0);
		val1 = TransformByMatrix(v1);
		val2 = TransformByMatrix(v2);
	}
	else
	{
		val0 = Transform(cameraPos, cameraDirection, v0);
		val1 = Transform(cameraPos, cameraDirection, v1);
		val2 = Transform(cameraPos, cameraDirection, v2);
	}
	

	DrawTriangle(val0, val1, val2, color);
}

void CpuGraphics::DrawRect(const Vec2f& v0, const Vec2f& v1, const Vec2f& v2, const Vec2f& v3, unsigned int color)
{
	DrawTriangle(v0, v1, v2, color);
	DrawTriangle(v0, v2, v3, color);
}

void CpuGraphics::DrawCrosshair()
{
	int offset = 10;
	for (int y = bufferHeight / 2 - offset; y < bufferHeight / 2 + offset; y++)
	{
		for (int x = bufferWidth / 2 - offset; x < bufferWidth / 2 + offset; x++)
		{
			screenBuffer[y * bufferWidth + x] = 0x0000FF;
		}
	}
}

Vec2f CpuGraphics::Transform(const Vec3f& camPos, const Vec3f& cam, const Vec3f& vertex) const
{
	Vec3f camToVert = vertex - camPos;
	auto projScale = cameraDirection.Dot(camToVert) / (fovLen * fovLen);
	Vec3f vertProj = cam * projScale;
	
	Vec3f vertProjPos = camPos + vertProj;
	Vec3f vertDir = vertex - vertProjPos;

	float projAng = acos(vertDir.Cos(cameraY90));
	float crossCamX = cameraY90.y * vertDir.z - cameraY90.z * vertDir.y;
	projAng = MathHelper::PI_2_f + ((crossCamX > 0) != (cam.x > 0) ? projAng : -projAng);
	
	float resultScale = vertDir.Length() / projScale;
	return Vec2f((bufferWidth >> 1) - cos(projAng) * resultScale, (bufferHeight >> 1) + sin(projAng) * resultScale);
}

Vec2f CpuGraphics::TransformByMatrix(const Vec3f& vertex) const
{
	Vec3f camToVert = vertex - cameraPos;
	auto projScale = cameraDirection.Dot(camToVert) / (fovLen * fovLen);

	auto res = transform.Mult(camToVert);
	return Vec2f((bufferWidth >> 1) + res.x / projScale, (bufferHeight >> 1) + res.y / projScale);
}