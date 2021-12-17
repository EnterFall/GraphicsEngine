#pragma once
#include "EFWin.h"
#include "EFException.h"
#include "DirectXGraphics.h"
#include "Vec2f.h"
#include "CpuGraphics.h"
#include "Keyboard.h"

#define WindowExcept(errCode) Window::Exception(__LINE__, __FILE__, errCode)
#define WindowExceptLastError() Window::Exception(__LINE__, __FILE__, GetLastError())
class Window
{
public:
	int height;
	int width;
	CpuGraphics graphics;
	std::unique_ptr<DirectXGraphics> dxGraphics;
	Keyboard keyboard;
private:
	HWND hWnd;
	HDC hdc;
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
	class StaticWindowClass
	{
	private:
		static StaticWindowClass wndClass;
		static constexpr const char* className = "EFWindowEngine";
		HINSTANCE hInstance;
	public:
		static HINSTANCE GetInstance();
		static LPCSTR GetName();
	private:
		StaticWindowClass();
		~StaticWindowClass();
		StaticWindowClass(const StaticWindowClass&) = delete;
		StaticWindowClass& operator =(const StaticWindowClass&) = delete;
	};
public:
	Window(int width, int height, std::string title);
	~Window();
	HWND GetHWnd() const;
	CpuGraphics& GetGraphics();
	void SetTitle(const std::string& newTitle) const;
	void UpdateScreen();
	bool ProcessMessages();
	Window(const Window&) = delete;
	Window& operator =(const Window&) = delete;
private:
	static LRESULT WINAPI RedirectMsgSetup(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);
	static LRESULT WINAPI RedirectMsg(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);
	LRESULT HandleMsg(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);
	void OnSize();
};