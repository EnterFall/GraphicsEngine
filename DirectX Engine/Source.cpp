#include "Window.h"
#include <chrono>
#include "Matrix3.h"

int WINAPI WinMain(
	HINSTANCE hInstance,
	HINSTANCE hPrevInstance,
	LPSTR lpCmdLine,
	int nCmdShow)
{
	try
	{
		Window w1(400, 200, "LolTestLol");

		double a = 0;
		while (true)
		{
			MSG msg;

			auto s = std::chrono::high_resolution_clock::now();
			
			while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE))
			{
				if (msg.message == WM_QUIT)
				{
					return 0;
				}
				TranslateMessage(&msg);
				DispatchMessage(&msg);
			}
			
			float speed = w1.graphics.travelSpeed;
			if (w1.keyboard.IsPressed('R'))
			{
				speed /= 10.0f;
			}
			if (w1.keyboard.IsPressed('B'))
			{
#ifdef _DEBUG
				DebugBreak();
#endif
			}
			if (w1.keyboard.IsPressed('W'))
			{
				w1.graphics.camera.pos = w1.graphics.camera.pos + (w1.graphics.camera.Z90 * speed);
			}
			if (w1.keyboard.IsPressed('S'))
			{
				w1.graphics.camera.pos = w1.graphics.camera.pos - (w1.graphics.camera.Z90 * speed);
			}
			if (w1.keyboard.IsPressed('D'))
			{
				w1.graphics.camera.pos = w1.graphics.camera.pos + (w1.graphics.camera.X90 * speed);
			}
			if (w1.keyboard.IsPressed('A'))
			{
				w1.graphics.camera.pos = w1.graphics.camera.pos - (w1.graphics.camera.X90 * speed);
			}
			if (w1.keyboard.IsPressed(VK_SPACE))
			{
				w1.graphics.camera.pos = w1.graphics.camera.pos + (w1.graphics.camera.Y90 * speed);
			}
			if (w1.keyboard.IsPressed(VK_SHIFT))
			{
				w1.graphics.camera.pos = w1.graphics.camera.pos - (w1.graphics.camera.Y90 * speed);
			}
			if (w1.keyboard.IsPressed('T'))
			{
				w1.graphics.isMatrixTransform = !w1.graphics.isMatrixTransform;
				Sleep(80);
			}

			POINT e;
			if (!GetCursorPos(&e))
				throw WindowExceptLastError();

			float x = (e.x - w1.graphics.widthHalf) / (float)w1.graphics.bufferHeight * MathHelper::PIMul2_f;
			float y = -(e.y - w1.graphics.heightHalf) / (float)w1.graphics.bufferHeight * MathHelper::PIMul2_f;

			w1.graphics.camera.Rotate(x, y);

			w1.graphics.Clear();
			for (size_t i = 0; i < 1; i++)
			{
				w1.graphics.DrawTriangle({ 0.0f, 0.0f, 500.0f }, { 0.0f, 200.0f, 500.0f }, { 5000.0f, 100.0f, 500.0f }, Colors::Red);

				w1.graphics.DrawCube({ 0.0f, 0.0f, 0.0f }, { 1000.0f, 1000.0f, 1000.0f }, Colors::White);

				//w1.graphics.DrawPoligon({ { 0.0f, 0.0f }, { 100.0f, 100.0f }, { 200.0f, 300.0f }, { 300.0f, 600.0f }, { 0.0f, 700.0f } }, 0xFFFFFF);
			}
			
			w1.graphics.DrawCrosshair();
			
			w1.UpdateScreen();
			w1.graphics.zBuffer.Clear();
			auto e1 = std::chrono::high_resolution_clock::now();
			auto fps = 1.0 / std::chrono::duration<double>(e1 - s).count();
			if (fps > 1000)
			{
				Sleep(10);
			}
			w1.SetTitle(std::to_string(fps) + " isMatrix: " + std::to_string(w1.graphics.isMatrixTransform));
			a += 0.1;
		};
	}
	catch (EFException& e)
	{
		MessageBox(nullptr, e.what(), "EF EXCEPTION", MB_OK | MB_ICONERROR);
	}
	catch (std::exception& e)
	{
		MessageBox(nullptr, e.what(), "Standart Exception", MB_OK | MB_ICONERROR);
	}
	catch (...)
	{
		MessageBox(nullptr, "???", "Unknown Exception", MB_OK | MB_ICONERROR);
	}
	return 0;
}
