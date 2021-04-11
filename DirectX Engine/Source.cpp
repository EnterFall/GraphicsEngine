#include "Window.h"

int WINAPI WinMain(
	HINSTANCE hInstance,
	HINSTANCE hPrevInstance,
	LPSTR lpCmdLine,
	int nCmdShow)
{
	try
	{
		Window w1(400, 200, "LolTestLol");

		MSG msg;
		BOOL result;
		while ((result = GetMessage(&msg, nullptr, 0, 0)) > 0)
		{
			TranslateMessage(&msg);
			DispatchMessage(&msg);
		}

		if (result == -1)
			return -1;
		else
			return msg.wParam;
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
