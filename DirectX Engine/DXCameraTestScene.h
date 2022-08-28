#pragma once
#include <thread> 
#include <DirectXMath.h>
#include "MathHelper.h"
#include "Window.h"

namespace dx = DirectX;
class DXCameraTestScene
{
private:
	DirectXGraphics* g;
	Keyboard* kbd;
	Window* w;
	dx::XMVECTOR camV;
	dx::XMVECTOR dir;
	dx::XMVECTOR dirUp;

	dx::XMVECTOR defZ;
	dx::XMVECTOR defY;
	dx::XMVECTOR defX;

	DXCamera camera;
	bool isFreeCamera;
public:
	double time = 0;
	DXCameraTestScene(Window* w, DirectXGraphics* g, Keyboard* kbd) : g(g), kbd(kbd)
	{
		srand(3454);
		camera = DXCamera();
		this->w = w;
	}

	void Play()
	{
		float speed = 0.02f;
		if (kbd->IsPressed('R'))
		{
			speed /= 10.0f;
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
			camera.pos = camera.pos + (camera.direction * speed);
		}
		if (kbd->IsPressed('S'))
		{
			camera.pos = camera.pos - (camera.direction * speed);
		}
		if (kbd->IsPressed('D'))
		{
			camera.pos = camera.pos + (camera.right * speed);
		}
		if (kbd->IsPressed('A'))
		{
			camera.pos = camera.pos - (camera.right * speed);
		}
		if (kbd->IsPressed(VK_SPACE))
		{
			camera.pos = camera.pos + (camera.up * speed);
		}
		if (kbd->IsPressed(VK_SHIFT))
		{
			camera.pos = camera.pos - (camera.up * speed);
		}

		float x = 0;
		float y = 0;

		POINT e = {};
		if (!GetCursorPos(&e))
			throw WindowExceptLastError();
		if (ScreenToClient(w->GetHWnd(), &e))
		{
			if (isFreeCamera)
			{
				x = (e.x - w->width / 2) / 300.0f;
				y = (e.y - w->height / 2) / 300.0f;

				camera.Rotate(y, x);

			}
		}
		




		auto start = std::chrono::high_resolution_clock::now();

		

		g->Clear(0, 0, 0);
		g->DrawTestTriangle(camera, y, x, camera.view);
		g->EndFrame();

		time = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count();
	}

	void Dispose()
	{

	}
};