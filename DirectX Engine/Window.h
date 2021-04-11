#pragma once
#include "EFWin.h"
#include "EFException.h"
#include <sstream>

#define EFWndExcept(errCode) Window::Exception(__LINE__, __FILE__, errCode)
#define EFWndExceptLastError() Window::Exception(__LINE__, __FILE__, GetLastError())
class Window
{
private:
	HWND hWnd;

public:
	class Exception : public EFException
	{
	private:
		HRESULT errCode;
	public:
		static std::string TranslateErrorCode(HRESULT errCode);

		Exception(int line, std::string file, HRESULT errCode);
		HRESULT GetErrorCode() const;
		std::string GetErrorString() const;

		const char* what() const override;
		std::string GetType() const override;
	};
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
	HWND GetHWnd();
	Window(const Window&) = delete;
	Window& operator =(const Window&) = delete;
private:
	static LRESULT WINAPI RedirectMsgSetup(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);
	static LRESULT WINAPI RedirectMsg(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);
	LRESULT HandleMsg(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);
};