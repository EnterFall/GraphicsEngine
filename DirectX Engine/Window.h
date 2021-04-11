#pragma once
#include "EFWin.h"
#include <sstream>

class Window
{
public:
	HWND hWnd;

private:
	class WindowClass
	{
	private:
		static WindowClass wndClass;
		static constexpr const char* className = "EFWindowEngine";
		HINSTANCE hInstance;
	public:
		static HINSTANCE GetInstance();
		static LPCSTR GetName();
	private:
		WindowClass();
		~WindowClass();
		WindowClass(const WindowClass&) = delete;
		WindowClass& operator =(const WindowClass&) = delete;
	};

public:
	Window(int width, int height, std::string title);
	~Window();
	Window(const Window&) = delete;
	Window& operator =(const Window&) = delete;
private:
	static LRESULT WINAPI RedirectMsgSetup(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);
	static LRESULT WINAPI RedirectMsg(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);
	LRESULT HandleMsg(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);
};