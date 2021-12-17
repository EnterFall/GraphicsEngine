#include "Window.h"
#include <memory>
#include <algorithm>

Window::StaticWindowClass Window::StaticWindowClass::wndClass = StaticWindowClass();

Window::StaticWindowClass::StaticWindowClass() : hInstance(GetModuleHandle(nullptr))
{
	WNDCLASSEX wc = { 0 };
	wc.cbSize = sizeof(wc);
	//wc.style = CS_OWNDC;
	wc.style = CS_HREDRAW | CS_VREDRAW;
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

	if (RegisterClassEx(&wc) == 0)
		throw WindowExceptLastError();
}

Window::StaticWindowClass::~StaticWindowClass() 
{
	UnregisterClass(className, hInstance);
}

HINSTANCE Window::StaticWindowClass::GetInstance()
{
	return wndClass.hInstance;
}

LPCSTR Window::StaticWindowClass::GetName()
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
	//auto styles = WS_CAPTION | WS_MINIMIZEBOX | WS_MAXIMIZEBOX | WS_SYSMENU;
	auto styles = WS_OVERLAPPEDWINDOW;

	if (!AdjustWindowRect(&rect, styles, false))
		throw WindowExceptLastError();

	hWnd = CreateWindowEx(
		0,
		StaticWindowClass::GetName(),
		title.c_str(),
		styles,
		CW_USEDEFAULT, CW_USEDEFAULT,
		width, height,
		nullptr,
		nullptr,
		StaticWindowClass::GetInstance(),
		this);
	if (hWnd == NULL)
		throw WindowExceptLastError();

	this->width = width;
	this->height = height;
	this->hdc = GetDC(hWnd);
	
	dxGraphics = std::make_unique<DirectXGraphics>(hWnd);

	// // - WS_VISIBLE is set
	ShowWindow(hWnd, SW_SHOWDEFAULT);

	
}

Window::~Window()
{
	if (!DestroyWindow(hWnd))
		throw WindowExceptLastError();
}

HWND Window::GetHWnd() const
{
	return hWnd;
}

CpuGraphics& Window::GetGraphics()
{
	return graphics;
}

void Window::SetTitle(const std::string& newTitle) const
{
	SetWindowText(hWnd, newTitle.c_str());
}

void Window::UpdateScreen()
{
	RECT rect;

	if (!GetClientRect(hWnd, &rect))
		throw WindowExceptLastError();

	int width = rect.right - rect.left;
	int height = rect.bottom - rect.top;
	this->width = width;
	this->height = height;

	POINT location{ 0, 0 };
	if (!ClientToScreen(hWnd, &location))
		throw WindowExceptLastError();

	if (!StretchDIBits(hdc, 0, 0, width, height, location.x, graphics.bufferHeight - location.y - height, width, height, graphics.GetScreenBuffer(), &graphics.GetBufferInfo(), DIB_RGB_COLORS, SRCCOPY))
		throw WindowExceptLastError();

	//dxGraphics->Clear(1.0f, 0.0f, 0.0f);
	//dxGraphics->TestDrawTriangle();

	//dxGraphics->EndFrame();
}

bool Window::ProcessMessages()
{
	MSG msg;
	while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE))
	{
		if (msg.message == WM_QUIT)
		{
			return false;
		}
		TranslateMessage(&msg);
		DispatchMessage(&msg);
	}
	return true;
}

void Window::OnSize()
{
	UpdateScreen();
}

LRESULT Window::RedirectMsgSetup(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
	if (msg == WM_NCCREATE)
	{
		auto pCreate = reinterpret_cast<CREATESTRUCT*>(lParam);
		auto pWnd = static_cast<Window*>(pCreate->lpCreateParams);
		SetWindowLongPtr(hWnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(pWnd));
		SetWindowLongPtr(hWnd, GWLP_WNDPROC, reinterpret_cast<LONG_PTR>(Window::RedirectMsg));
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
	switch (msg)
	{
	case WM_KEYDOWN:
		keyboard.KeyDownFromMessage(static_cast<char>(wParam), static_cast<char>(lParam));
		break;
	case WM_KEYUP:
		keyboard.KeyUpFromMessage(static_cast<char>(wParam), static_cast<char>(lParam));
		break;
	case WM_CHAR:
		break;
	case WM_KILLFOCUS:
		keyboard.Clear();
		break;
	case WM_CLOSE:
		PostQuitMessage(0);
		break;
	case WM_LBUTTONDOWN:
		break;
	case WM_SIZE:
		OnSize();
		break;
	case WM_SIZING:
		OnSize();
		return true;
	case WM_MOVING:
		OnSize();
		return true;
	default:
		return DefWindowProc(hWnd, msg, wParam, lParam);
	}

	return 0;
}

std::string Window::Exception::TranslateErrorCode(HRESULT errCode)
{
	char* pMesBuffer = nullptr;
	DWORD msgLen = FormatMessage(
		FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
		nullptr,
		errCode,
		MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
		reinterpret_cast<LPSTR>(&pMesBuffer),
		0,
		nullptr);
	if (msgLen == 0)
	{
		return "Unidentified error code";
	}
	std::string error = pMesBuffer;
	LocalFree(pMesBuffer);
	return error;
}

Window::Exception::Exception(int line, std::string file, HRESULT errCode)
	: EFException(line, file), errCode(errCode)
{}

HRESULT Window::Exception::GetErrorCode() const
{
	return errCode;
}

std::string Window::Exception::GetErrorString() const
{
	return TranslateErrorCode(errCode);
}

const char* Window::Exception::what() const
{
	std::ostringstream stream;
	stream << GetType() << std::endl
		<< GetString() << std::endl
		<< "[Error code] " << GetErrorCode() << std::endl
		<< "[Description] " << GetErrorString();
	whatBuffer = stream.str();
	return whatBuffer.c_str();
}

std::string Window::Exception::GetType() const
{
	return "EF Window Exception";
}
