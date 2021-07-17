#include "CpuGraphics.h"

CpuGraphics::CpuGraphics()
{
	screenBuffer = std::make_shared<Color[]>(bufferWidth * bufferHeight);
	clipBuffer = std::vector<Vec3d>(6);

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

void CpuGraphics::DrawScreenTriangle(const Vec3d& v0, const Vec3d& v1, const Vec3d& v2, Color color)
{
	const Vec3d* p0 = &v0;
	const Vec3d* p1 = &v1;
	const Vec3d* p2 = &v2;
	if (p0->y > p1->y) std::swap(p0, p1);
	if (p1->y > p2->y) std::swap(p1, p2);
	if (p0->y > p1->y) std::swap(p0, p1);

	if (p0->y != p1->y && p1->y != p2->y)
	{
		auto r = *p2 - *p0;
		Vec3d cross = { p0->x + r.x * ((p1->y - p0->y) / r.y), p1->y, p0->z + r.z * ((p1->y - p0->y) / r.y) };
		const Vec3d* c = &cross;

		if (c->x < p1->x) std::swap(c, p1);
		DrawTriangleFromTo(*p0, *p1, *p0, *c, color);
		DrawTriangleFromTo(*p1, *p2, *c, *p2, color);
	}
	else if (p1->y == p2->y)
	{
		if (p1->x > p2->x) std::swap(p1, p2);
		DrawTriangleFromTo(*p0, *p1, *p0, *p2, color);
	}
	else if (p0->y == p1->y)
	{
		if (p0->x > p1->x) std::swap(p0, p1);
		DrawTriangleFromTo(*p0, *p2, *p1, *p2, color);
	}
}

// Sometimes pixel holes appear (
void CpuGraphics::DrawTriangleFromTo(const Vec3d& leftS, const Vec3d& leftE, const Vec3d& rightS, const Vec3d& rightE, Color color)
{
	Vec3d vLeft = leftE - leftS;
	Vec3d vRight = rightE - rightS;

	double dxLeft = vLeft.x / vLeft.y;
	double dxRight = vRight.x / vRight.y;

	double dzLeft = vLeft.z / vLeft.y;

	double outLeft = std::clamp(-leftS.y / vLeft.y, 0.0, 1.0);
	double outRight = std::clamp(-rightS.y / vRight.y, 0.0, 1.0);

	int yStart = std::clamp((int)ceil(leftS.y - 0.5), 0, bufferHeight);
	int yEnd = std::clamp((int)ceil(rightE.y - 0.5), 0, bufferHeight);

	// if y < 0, start vectors will be corrected
	Vec3d leftCorr = leftS + vLeft * outLeft;
	Vec3d rightCorr = rightS + vRight * outRight;

	double t;
	double xLeft = leftCorr.x + dxLeft * modf(0.5 - leftCorr.y + double(yStart), &t);
	double xRight = rightCorr.x + dxRight * modf(0.5 - rightCorr.y + double(yStart), &t);

	double zLeft = leftCorr.z + dzLeft * modf(0.5 - leftCorr.y + double(yStart), &t);

	Vec3d d = ((rightE + rightS) * 0.5) - ((leftE + leftS) * 0.5);
	double dz = d.z / d.x;
	int index = yStart * bufferWidth;
	for (int y = yStart; y < yEnd; y++)
	{
		int xStart = std::clamp((int)ceil(xLeft - 0.5), 0, bufferWidth);
		int xEnd = std::clamp((int)ceil(xRight - 0.5), 0, bufferWidth);
		
		//generates small artifacts on crossing
		//double z = zLeft + dz * (std::clamp(xLeft, 0.0, (double)bufferWidth) - xLeft + (modf(0.5 - xLeft, &t)));
		//double z = zLeft + dz * (xStart - xLeft + abs(xLeft - ceil(xLeft - 0.5)));
		double z = zLeft + dz * (std::clamp(xLeft, 0.0, (double)bufferWidth) - xLeft);
		for (int i = xStart; i < xEnd; i++)
		{
			if (zBuffer.Update(index + i, 1.0 / z))
			{
				screenBuffer[index + i] = color;
			}
			z += dz;
		}
		xLeft += dxLeft;
		xRight += dxRight;
		zLeft += dzLeft;
		index += bufferWidth;
	}
}

void CpuGraphics::DrawTriangle(const Vec3d& v0, const Vec3d& v1, const Vec3d& v2, Color color)
{
	// Backface culling, triangle in clockwise order is visible
	if ((v1 - v0).Cross(v2 - v0).Dot(v0 - camera.pos) < 0.0)
	{
		Vec3d val0;
		Vec3d val1;
		Vec3d val2;
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
		if (val0.z < zClip || val1.z < zClip || val2.z < zClip)
		{
			Clip(&clipBuffer, val0, val1);
			Clip(&clipBuffer, val1, val2);
			Clip(&clipBuffer, val2, val0);

			DrawPoligon(clipBuffer.begin()._Ptr, clipBuffer.size(), color);
			clipBuffer.clear();
		}
		else
		{
			DrawScreenTriangle(camera.ToScreen(val0), camera.ToScreen(val1), camera.ToScreen(val2), color);
		}
	}
}

// Only Z clipping
void CpuGraphics::Clip(std::vector<Vec3d>* list, const Vec3d& v0, const Vec3d& v1)
{
	const Vec3d* p0 = &v0;
	const Vec3d* p1 = &v1;

	if (p0->z < zClip && p1->z > zClip)
	{
		double scale = (zClip - p0->z) / (p1->z - p0->z);
		auto corr = *p0 + ((*p1 - *p0) * scale);

		list->push_back(camera.ToScreen(corr));
		list->push_back(camera.ToScreen(*p1));
	}
	else if (p0->z > zClip && p1->z < zClip)
	{
		double scale = (zClip - p1->z) / (p0->z - p1->z);
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

void CpuGraphics::ClipFull(std::vector<Vec3d>* list, const Vec3d& v0, const Vec3d& v1)
{
	double zClip = 0.1;

	const Vec3d* p0 = &v0;
	const Vec3d* p1 = &v1;

	if (p0->z < zClip && p1->z > zClip)
	{
		double scale = (zClip - p0->z) / (p1->z - p0->z);
		auto corr = *p0 + ((*p1 - *p0) * scale);

		list->push_back(camera.ToScreen(corr));
		list->push_back(camera.ToScreen(*p1));
	}
	else if (p0->z > zClip && p1->z < zClip)
	{
		double scale = (zClip - p1->z) / (p0->z - p1->z);
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

void CpuGraphics::DrawRect(const Vec3d& v0, const Vec3d& v1, const Vec3d& v2, const Vec3d& v3, Color color)
{
	DrawScreenTriangle(v0, v1, v2, color);
	DrawScreenTriangle(v0, v2, v3, color);
}

// Not optimized
void CpuGraphics::DrawCube(const Vec3d& p0, const Vec3d& p1, Color color)
{
	auto dif = p1 - p0;
	auto x = Vec3d(dif.x, 0.0f, 0.0f);
	auto y = Vec3d(0.0f, dif.y, 0.0f);
	auto z = Vec3d(0.0f, 0.0f, dif.z);
	
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

void CpuGraphics::DrawPoligon(Vec3d* points, size_t count, Color color)
{
	auto v0 = points;
	auto v1 = points + 1;
	auto v2 = points + 2;
	int i = 2;
	while (i < count)
	{
		if (*v1 != *v2)
		{
			DrawScreenTriangle(*v0, *v1, *v2, color);
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

void CpuGraphics::Draw(const Cube& cube, Color color)
{
	const Vec3d* v = cube.GetVerts();
	DrawTriangle(v[0], v[2], v[1], color);
	DrawTriangle(v[1], v[2], v[3], color);
	DrawTriangle(v[0], v[4], v[2], color);
	DrawTriangle(v[4], v[6], v[2], color);
	DrawTriangle(v[1], v[3], v[5], color);
	DrawTriangle(v[5], v[3], v[7], color);
	DrawTriangle(v[0], v[1], v[4], color);
	DrawTriangle(v[4], v[1], v[5], color);

	DrawTriangle(v[2], v[6], v[7], color);
	DrawTriangle(v[7], v[3], v[2], color);
	DrawTriangle(v[4], v[5], v[7], color);
	DrawTriangle(v[7], v[6], v[4], color);
}

Vec3d CpuGraphics::Transform(const Vec3d& vertex) const
{
	Vec3d camToVert = vertex - camera.pos;

	auto projX = camera.X90.Dot(camToVert);
	auto projY = camera.Y90.Dot(camToVert);
	auto projZ = camera.Z90.Dot(camToVert);

	return Vec3d(projX, projY, projZ);
}

Vec3d CpuGraphics::TransformByMatrix(const Vec3d& vertex) const
{
	Vec3d camToVert = vertex - camera.pos;
	return camera.transform.Mult(camToVert);
}