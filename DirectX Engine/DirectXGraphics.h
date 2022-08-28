#pragma once
#include <wrl.h>
#include <d3d11.h>
#include <string>
#include <comdef.h>
#include <DirectXMath.h>
#include "EFWin.h"
#include "EFException.h"
#include "Camera.h"
#include "DXCamera.h"

class DXCodeError
{
public:
	static std::string DXTranslateErrorCode(HRESULT errCode)
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

	static void DXAssert_Impl(int line, std::string file, HRESULT code)
	{
		int a = code;
		if (FAILED(code)) {

			_com_error err(code);
			LPCTSTR errMsg = err.ErrorMessage();
			throw EFException(line, file, errMsg);
		}
	}
};

#define DXAssert(code) DXCodeError::DXAssert_Impl(__LINE__, __FILE__, code)

class DirectXGraphics
{
private:
	Microsoft::WRL::ComPtr<ID3D11Device> device;
	Microsoft::WRL::ComPtr<IDXGISwapChain> swapChain;
	Microsoft::WRL::ComPtr<ID3D11DeviceContext> deviceContext;
	Microsoft::WRL::ComPtr<ID3D11RenderTargetView> targetView;
public:
	DirectXGraphics(HWND hwnd);
	void EndFrame();
	void DrawTestTriangle(const DXCamera& cam, float pitch, float yaw, DirectX::XMMATRIX view);
	void CreateSetVShaderAndIL(LPCWCHAR fileName);
	void CreateSetPShader(LPCWCHAR fileName);
	void ConstBufferVSSet(size_t bSize, void* dataFrom);
	void Clear(float r, float g, float b);
private:
	template <class T>
	void SetVertexBuffer(T* arr, size_t count);
};

