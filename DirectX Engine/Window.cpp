#include "Window.h"

Window::WindowClass Window::WindowClass::wndClass;

Window::WindowClass::WindowClass() : hInstance(GetModuleHandle(nullptr))
{
	WNDCLASSEX wc = { 0 };
	wc.cbSize = sizeof(wc);
	wc.style = CS_OWNDC;
	wc.lpfnWndProc = RedirectMsgSetup;
	wc.cbClsExtra = 0;
	wc.cbWndExtra = 0;
	wc.hInstance = hInstance;
	wc.hIcon = nullptr;
	wc.hCursor = nullptr;
	wc.hbrBackground = nullptr;
	wc.lpszMenuName = nullptr;
	wc.lpszClassName = className;
	wc.hIconSm = nullptr;

	RegisterClassEx(&wc);
}

Window::WindowClass::~WindowClass() {
	UnregisterClass(className, hInstance);
}

HINSTANCE Window::WindowClass::GetInstance()
{
	return wndClass.hInstance;
}

LPCSTR Window::WindowClass::GetName()
{
	return wndClass.className;
}

Window::Window(int width, int height, std::string title)
{
	RECT rect;
	rect.left = 100;
	rect.right = width + rect.left;
	rect.top = 100;
	rect.bottom = height + rect.top;

	auto styles = WS_CAPTION | WS_MINIMIZEBOX | WS_SYSMENU;

	AdjustWindowRect(&rect, styles, false);

	hWnd = CreateWindowEx(
		0,
		WindowClass::GetName(),
		title.c_str(),
		styles,
		CW_USEDEFAULT, CW_USEDEFAULT, 
		rect.right - rect.left, rect.bottom - rect.top,
		nullptr,
		nullptr,
		WindowClass::GetInstance(),
		this);

	SetWindowLongPtr(hWnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(this));
	SetWindowLongPtr(hWnd, GWLP_WNDPROC, reinterpret_cast<LONG_PTR>(&Window::RedirectMsg));
	ShowWindow(hWnd, SW_SHOWDEFAULT);
}

Window::~Window()
{
	DestroyWindow(hWnd);
}

LRESULT Window::RedirectMsgSetup(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
	if (msg == WM_NCCREATE)
	{
		auto pCreate = reinterpret_cast<CREATESTRUCT*>(lParam);
		auto pWnd = static_cast<Window*>(pCreate->lpCreateParams);
		SetWindowLongPtr(hWnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(pWnd));
		SetWindowLongPtr(hWnd, GWLP_WNDPROC, reinterpret_cast<LONG_PTR>(&Window::RedirectMsg));
		return pWnd->HandleMsg(hWnd, msg, wParam, lParam);
	}
	return DefWindowProc(hWnd, msg, wParam, lParam);
}

LRESULT Window::RedirectMsg(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
	auto pWnd = reinterpret_cast<Window*>(GetWindowLongPtr(hWnd, GWLP_USERDATA));
	return pWnd->HandleMsg(hWnd, msg, wParam, lParam);
}

LRESULT Window::HandleMsg(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
	static std::string title;
	switch (msg)
	{
	case WM_KEYDOWN:
		if (wParam == 'F')
		{
			SetWindowText(hWnd, "Respects!");
		}
		break;
	case WM_CHAR:
		title.push_back(wParam);
		SetWindowText(hWnd, title.c_str());
		break;
	case WM_CLOSE:
		PostQuitMessage(69);
		return 0;
	case WM_LBUTTONDOWN:
		auto p = MAKEPOINTS(lParam);
		std::ostringstream s;
		s << p.x << "  " << p.y;
		SetWindowText(hWnd, s.str().c_str());
		break;
	}

	return DefWindowProc(hWnd, msg, wParam, lParam);
}
