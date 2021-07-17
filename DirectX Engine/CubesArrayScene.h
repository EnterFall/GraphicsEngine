#pragma once
#include "Window.h"
#include "MathHelper.h"

class CubesArrayScene
{
private:
	CpuGraphics* g;
	Keyboard* kbd;
	std::vector<Cube> cubes;
	bool isFreeCamera;
	unsigned int size;
public:
	CubesArrayScene(CpuGraphics* g, Keyboard* kbd, unsigned int size) : g(g), kbd(kbd), size(size)
	{
		// 22 fps (40)
		srand(3454);
		g->camera.pos = Vec3d(-50.0, 50.0, -50.0) / 5;
		g->camera.Rotate(MathHelper::PI_4, -MathHelper::PI_4 + 0.2);
		cubes = std::vector<Cube>();
		for (unsigned int z = 0; z < size; z++)
		{
			for (int y = -(int)size; y < 0; y++)
			{
				for (unsigned int x = 0; x < size; x++)
				{
					if (rand() % 4 == 0)
					{
						cubes.emplace_back(Vec3d(x, y, z), 1.0);
					}
				}
			}
		}
	}

	void Play()
	{
		double speed = g->travelSpeed;
		if (kbd->IsPressed('R'))
		{
			speed /= 10.0;
		}
		if (kbd->IsPressed('C'))
		{
			isFreeCamera = !isFreeCamera;
			Sleep(80);
		}
		if (kbd->IsPressed('B'))
		{
#ifdef _DEBUG
			DebugBreak();
#endif
		}
		if (kbd->IsPressed('W'))
		{
			g->camera.pos = g->camera.pos + (g->camera.Z90 * speed);
		}
		if (kbd->IsPressed('S'))
		{
			g->camera.pos = g->camera.pos - (g->camera.Z90 * speed);
		}
		if (kbd->IsPressed('D'))
		{
			g->camera.pos = g->camera.pos + (g->camera.X90 * speed);
		}
		if (kbd->IsPressed('A'))
		{
			g->camera.pos = g->camera.pos - (g->camera.X90 * speed);
		}
		if (kbd->IsPressed(VK_SPACE))
		{
			g->camera.pos = g->camera.pos + (g->camera.Y90 * speed);
		}
		if (kbd->IsPressed(VK_SHIFT))
		{
			g->camera.pos = g->camera.pos - (g->camera.Y90 * speed);
		}
		if (kbd->IsPressed('T'))
		{
			g->isMatrixTransform = !g->isMatrixTransform;
			Sleep(80);
		}

		POINT e;
		if (!GetCursorPos(&e))
			throw WindowExceptLastError();

		if (isFreeCamera)
		{
			float x = (e.x - g->widthHalf) / (float)g->bufferHeight * MathHelper::PI2_f;
			float y = -(e.y - g->heightHalf) / (float)g->bufferHeight * MathHelper::PI2_f;

			g->camera.Rotate(x, y);
		}

		g->Clear();

		double scaler = 256.0 / size;
		for (size_t i = 0; i < cubes.size(); i++)
		{
			g->Draw(cubes[i], Color(cubes[i].verts[0].x * scaler, cubes[i].verts[0].y * scaler, cubes[i].verts[0].z * scaler));
		}

	}
};