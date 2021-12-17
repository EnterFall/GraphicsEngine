#include "DirectXGraphics.h"
#include "Vec2f.h"
#include <vector>
#include <d3dcompiler.h>

#pragma comment(lib,"d3d11.lib")
#pragma comment(lib,"D3DCompiler.lib")

DirectXGraphics::DirectXGraphics(HWND hwnd)
{
	DXGI_SWAP_CHAIN_DESC desc = { 0 };
	desc.BufferDesc.Width = 0u;
	desc.BufferDesc.Height = 0u;
	desc.BufferDesc.RefreshRate.Numerator = 0;
	desc.BufferDesc.RefreshRate.Denominator = 1;
	desc.BufferDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
	desc.BufferDesc.ScanlineOrdering = DXGI_MODE_SCANLINE_ORDER_UNSPECIFIED;
	desc.BufferDesc.Scaling = DXGI_MODE_SCALING_UNSPECIFIED;
	desc.SampleDesc.Count = 1;
	desc.SampleDesc.Quality = 0;
	desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
	desc.BufferCount = 1;
	desc.OutputWindow = hwnd;
	desc.Windowed = TRUE;
	desc.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;

	D3D_FEATURE_LEVEL	featureLevelsRequested = D3D_FEATURE_LEVEL_11_0;
	UINT				numLevelsRequested = 1;
	D3D_FEATURE_LEVEL	featureLevelsSupported;

	DXAssert(D3D11CreateDeviceAndSwapChain(
		nullptr,
		D3D_DRIVER_TYPE_HARDWARE,
		nullptr,
		D3D11_CREATE_DEVICE_DEBUG,
		&featureLevelsRequested,
		numLevelsRequested,
		D3D11_SDK_VERSION,
		&desc,
		&swapChain,
		&device,
		&featureLevelsSupported,
		&deviceContext
	));

	Microsoft::WRL::ComPtr<ID3D11Resource> res;
	DXAssert(swapChain->GetBuffer(0u, __uuidof(ID3D11Resource), &res));
	device->CreateRenderTargetView(res.Get(), nullptr, &targetView);


	deviceContext->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
}

void DirectXGraphics::EndFrame()
{
	DXAssert(swapChain->Present(0u, 0u));
}

void DirectXGraphics::TestDrawTriangle()
{
	struct Vertex
	{
		float x;
		float y;
		float z;
	};
	std::vector<Vertex> verts = std::vector<Vertex>();
	verts.emplace_back(0.0f, 0.0f, 0.0f);
	verts.emplace_back(0.5f, -0.5f, 0.0f);
	verts.emplace_back(-0.5f, -0.5f, 0.0f);

	D3D11_BUFFER_DESC desc = { };
	desc.ByteWidth = verts.size() * sizeof(Vertex);
	desc.Usage = D3D11_USAGE_DEFAULT;
	desc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
	desc.CPUAccessFlags = 0u;
	desc.MiscFlags = 0u;
	desc.StructureByteStride = sizeof(Vertex);
	D3D11_SUBRESOURCE_DATA data = { };
	data.pSysMem = verts.begin()._Ptr;
	Microsoft::WRL::ComPtr<ID3D11Buffer> vertBuffer;
	DXAssert(device->CreateBuffer(&desc, &data, &vertBuffer));
	UINT strides = sizeof(Vertex);
	UINT offsets = 0u;
	deviceContext->IASetVertexBuffers(0u, 1u, vertBuffer.GetAddressOf(), &strides, &offsets);

	Microsoft::WRL::ComPtr<ID3D11VertexShader> vertShader;
	Microsoft::WRL::ComPtr<ID3DBlob> blob;
	DXAssert(D3DReadFileToBlob(L"VertexShader.cso", &blob));
	DXAssert(device->CreateVertexShader(blob->GetBufferPointer(), blob->GetBufferSize(), nullptr, &vertShader));
	deviceContext->VSSetShader(vertShader.Get(), nullptr, 0u);

	const D3D11_INPUT_ELEMENT_DESC elDesc[] = 
	{
		{ "Position", 0u, DXGI_FORMAT_R32G32B32_FLOAT, 0u, 0u, D3D11_INPUT_PER_VERTEX_DATA, 0u },
	};
	Microsoft::WRL::ComPtr<ID3D11InputLayout> inputLayout;
	DXAssert(device->CreateInputLayout(elDesc, std::size(elDesc), blob->GetBufferPointer(), blob->GetBufferSize(), &inputLayout));
	deviceContext->IASetInputLayout(inputLayout.Get());

	Microsoft::WRL::ComPtr<ID3D11PixelShader> pixelShader;
	DXAssert(D3DReadFileToBlob(L"PixelShader.cso", &blob));
	DXAssert(device->CreatePixelShader(blob->GetBufferPointer(), blob->GetBufferSize(), nullptr, &pixelShader));
	deviceContext->PSSetShader(pixelShader.Get(), nullptr, 0u);

	deviceContext->OMSetRenderTargets(1u, targetView.GetAddressOf(), nullptr);

	D3D11_VIEWPORT viewPort = { };
	viewPort.Width = 1000.0f;
	viewPort.Height = 500.0f;
	viewPort.MinDepth = 0.0f;
	viewPort.MaxDepth = 1.0f;
	viewPort.TopLeftX = 0.0f;
	viewPort.TopLeftY = 0.0f;
	deviceContext->RSSetViewports(1u, &viewPort);

	deviceContext->Draw((UINT)verts.size(), 0u);
}

void DirectXGraphics::Clear(float r, float g, float b)
{
	float color[] = { r, g, b, 1.0f };
	deviceContext->ClearRenderTargetView(targetView.Get(), color);
}
