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
			
			if (w1.keyboard.IsPressed('B'))
			{
#ifdef _DEBUG
				DebugBreak();
#endif
			}
			if (w1.keyboard.IsPressed('W'))
			{
				w1.graphics.camera.pos = w1.graphics.camera.pos + (w1.graphics.camera.Z90 * w1.graphics.travelSpeed);
			}
			if (w1.keyboard.IsPressed('S'))
			{
				w1.graphics.camera.pos = w1.graphics.camera.pos - (w1.graphics.camera.Z90 * w1.graphics.travelSpeed);
			}
			if (w1.keyboard.IsPressed('D'))
			{
				w1.graphics.camera.pos = w1.graphics.camera.pos + (w1.graphics.camera.X90 * w1.graphics.travelSpeed);
			}
			if (w1.keyboard.IsPressed('A'))
			{
				w1.graphics.camera.pos = w1.graphics.camera.pos - (w1.graphics.camera.X90 * w1.graphics.travelSpeed);
			}
			if (w1.keyboard.IsPressed(VK_SHIFT))
			{
				w1.graphics.camera.pos = w1.graphics.camera.pos + (w1.graphics.camera.Y90 * w1.graphics.travelSpeed);
			}
			if (w1.keyboard.IsPressed(VK_SPACE))
			{
				w1.graphics.camera.pos = w1.graphics.camera.pos - (w1.graphics.camera.Y90 * w1.graphics.travelSpeed);
			}


			if (w1.keyboard.IsPressed('T'))
			{
				w1.graphics.isMatrixTransform = !w1.graphics.isMatrixTransform;
				Sleep(80);
			}


			POINT e;
			if (!GetCursorPos(&e))
				throw WindowExceptLastError();

			float x = (e.x - w1.graphics.bufferWidth / 2) / (float)w1.graphics.bufferHeight * MathHelper::PIMul2_f;
			float y = (e.y - w1.graphics.bufferHeight / 2) / (float)w1.graphics.bufferHeight * MathHelper::PIMul2_f;

			w1.graphics.camera.Rotate(x, y);

			//w1.DrawRect(100, 100, 200, 200, (unsigned int)a % 256);
			w1.graphics.Clear();
			for (size_t i = 0; i < 1; i++)
			{
				// i = [0..128] 520
				//g.DrawTriangle({ 100.0f, 250.0f }, { 576.5f, 1000.5f }, { 445.0f, 700.0f }, 0xFFFFFF);
				
				// i = [0..128] 58
				//g.DrawTriangle({ -10000.0f, -10000.0f }, { 2000.0f, 10000.0f }, { 2000.0f, -100.0f }, 0xFFFFFF);

				//g.DrawTriangle({ 2500.0f, -150.0f }, { 576.5f, 200.5f }, { 500.0f, 500.0f }, 0xFFFFFF);
				//g.DrawTriangle({ 2500.0f, 1400.0f }, { 1076.5f, 200.5f }, { 200.5f, 576.5f }, 0xFFFFFF);
				//g.DrawTriangle({ -100.0f, -500.0f }, { -500.0f, -100.0f }, { 300.0f, 300.0f }, 0xFFFFFF);
				
				//w1.graphics.DrawTriangle({ 0.0f, 1080.0f / 2, 500.0f }, { 0.0f, 0.0f, 500.0f }, { 1920.0f / 2, 0.0f, 500.0f }, 0xFFFFFF);
				//w1.graphics.DrawTriangle({ 0.0f, 1080.0f / 2, 500.0f }, { 1920.0f / 2, 0.0f, 500.0f }, { 1920.0f / 2, 1080.0f / 2, 500.0f }, 0xFFFFFF);

				w1.graphics.DrawTriangle({ 0.0f, 0.0f, 500.0f }, { 200.0f, 100.0f, 500.0f }, { 0.0f, 200.0f, 500.0f }, 0xFFFFFF);

				//w1.graphics.DrawCube({ 0.0f, 0.0f, 0.0f }, { 100.0f, 100.0f, 100.0f }, 0xFFFFFF);

				//w1.graphics.DrawPoligon({ { 0.0f, 0.0f }, { 100.0f, 100.0f }, { 200.0f, 300.0f }, { 300.0f, 600.0f }, { 0.0f, 700.0f } }, 0xFFFFFF);
			}
			
			w1.graphics.DrawCrosshair();
			
			w1.UpdateScreen();
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
