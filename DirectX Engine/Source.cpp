#include "Window.h"
#include <chrono>
#include "Matrix3.h"
#include "CubesArrayScene.h"

int WINAPI WinMain(
	HINSTANCE hInstance,
	HINSTANCE hPrevInstance,
	LPSTR lpCmdLine,
	int nCmdShow)
{
	try
	{
		Window w1(1900, 1000, "LolTestLol");

		auto scene = CubesArrayScene(&w1.graphics, &w1.keyboard, 40);
		while (true)
		{
			auto s = std::chrono::high_resolution_clock::now();
			
			if (!w1.ProcessMessages())
			{
				return 0;
			}

			scene.Play();

			w1.UpdateScreen();
			w1.graphics.zBuffer.Clear();

			auto e1 = std::chrono::high_resolution_clock::now();
			auto fps = 1.0 / std::chrono::duration<double>(e1 - s).count();
			if (fps > 1000)
			{
				Sleep(10);
			}
			w1.SetTitle(std::to_string(fps) + " isMatrix: " + std::to_string(w1.graphics.isMatrixTransform));
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
		MessageBox(nullptr, "what?", "Unknown Exception???", MB_OK | MB_ICONERROR);
	}
	return 0;
}
