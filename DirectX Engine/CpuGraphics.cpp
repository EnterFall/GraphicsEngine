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

void CpuGraphics::DrawTriangleFlatTop(const Vec2f& v0, const Vec2f& v1, const Vec2f& v2, unsigned int color)
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

void CpuGraphics::DrawTriangle(const Vec3f& v0, const Vec3f& v1, const Vec3f& v2, unsigned int color)
{
	// Backface culling
	if ((v1 - v0).Cross(v2 - v0).Dot(v0 - cameraPos) < 0)
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
}

void CpuGraphics::DrawRect(const Vec2f& v0, const Vec2f& v1, const Vec2f& v2, const Vec2f& v3, unsigned int color)
{
	DrawTriangle(v0, v1, v2, color);
	DrawTriangle(v0, v2, v3, color);
}

void CpuGraphics::DrawCube(const Vec3f& p0, const Vec3f& p1, unsigned int color)
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

void CpuGraphics::DrawPoligon(const std::initializer_list<Vec2f>& points, unsigned int color)
{
	auto pointsArr = points.begin();
	auto v0 = pointsArr;
	auto v1 = pointsArr + 1;
	auto v2 = pointsArr + 2;
	int size = points.size();
	int i = 2;
	while (i < size)
	{
		DrawTriangle(*v0, *v1, *v2, color);
		v1 = v2;
		v2 = pointsArr + 1 + i;
		i++;
	}
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

// Z value is not computed
Vec2f CpuGraphics::Transform(const Vec3f& camPos, const Vec3f& cam, const Vec3f& vertex) const
{
	Vec3f camToVert = vertex - camPos;

	// projScale = (camToVert.Length() * cameraDirection.Cos(camToVert)) / cameraDirection.Length();
	auto projScale = cameraDirection.Dot(camToVert) * fovScalar;

	auto projX = cameraX90.Dot(camToVert);
	auto projY = cameraY90.Dot(camToVert);

	return Vec2f((bufferWidth >> 1) + projX / projScale, (bufferHeight >> 1) + projY / projScale);
}

Vec2f CpuGraphics::TransformByMatrix(const Vec3f& vertex) const
{
	Vec3f camToVert = vertex - cameraPos;
	auto projScale = cameraDirection.Dot(camToVert) * fovScalar;

	auto proj = transform.Mult(camToVert) / projScale;
	return Vec2f((bufferWidth >> 1) + proj.x, (bufferHeight >> 1) + proj.y);
}