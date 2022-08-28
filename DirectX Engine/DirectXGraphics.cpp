#include "DirectXGraphics.h"
#include "Vec2f.h"
#include <vector>
#include <d3dcompiler.h>


using namespace DirectX;

#pragma comment(lib,"d3d11.lib")
#pragma comment(lib,"D3DCompiler.lib")

DirectXGraphics::DirectXGraphics(HWND hwnd)
{
	DXGI_SWAP_CHAIN_DESC desc = {};
	desc.BufferDesc.Width = 0u;
	desc.BufferDesc.Height = 0u;
	desc.BufferDesc.RefreshRate.Numerator = 0;
	desc.BufferDesc.RefreshRate.Denominator = 1;
	desc.BufferDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
	desc.BufferDesc.ScanlineOrdering = DXGI_MODE_SCANLINE_ORDER_UNSPECIFIED;
	desc.BufferDesc.Scaling = DXGI_MODE_SCALING_UNSPECIFIED;
	desc.SampleDesc.Count = 8;
	desc.SampleDesc.Quality = 0;
	desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
	desc.BufferCount = 1;
	desc.OutputWindow = hwnd;
	desc.Windowed = TRUE;
	desc.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;

	D3D_FEATURE_LEVEL	featureLevelsRequested = D3D_FEATURE_LEVEL_11_1;
	UINT				numLevelsRequested = 1;
	D3D_FEATURE_LEVEL	featureLevelsSupported;

	DXAssert(D3D11CreateDeviceAndSwapChain(
		nullptr,
		D3D_DRIVER_TYPE_HARDWARE,
		nullptr,
		D3D11_CREATE_DEVICE_FLAG::D3D11_CREATE_DEVICE_DEBUG,
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
	DXAssert(device->CreateRenderTargetView(res.Get(), nullptr, &targetView));
}

void DirectXGraphics::EndFrame()
{
	DXAssert(swapChain->Present(0u, 0u));
}

void DirectXGraphics::DrawTestTriangle(const DXCamera& camera, float pitch, float yaw, DirectX::XMMATRIX view)
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
	verts.emplace_back(-0.5f, -0.8f, 0.0f);
	//verts.emplace_back(-0.2f, -1.0f, 0.0f);
	//verts.emplace_back(-0.5f, -0.8f, 0.0f);

	D3D11_BUFFER_DESC desc = {};
	desc.ByteWidth = verts.size() * sizeof(Vertex);
	desc.Usage = D3D11_USAGE_DEFAULT;
	desc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
	desc.CPUAccessFlags = 0u;
	desc.MiscFlags = 0u;
	desc.StructureByteStride = sizeof(Vertex);
	D3D11_SUBRESOURCE_DATA data = {};
	data.pSysMem = verts.data();
	Microsoft::WRL::ComPtr<ID3D11Buffer> vertBuffer;
	DXAssert(device->CreateBuffer(&desc, &data, &vertBuffer));
	UINT stride = sizeof(Vertex);
	UINT offset = 0u;
	deviceContext->IASetVertexBuffers(0u, 1u, vertBuffer.GetAddressOf(), &stride, &offset);

	CreateSetVShaderAndIL(L"VertexShader.cso");
	CreateSetPShader(L"PixelShader.cso");




	


	deviceContext->OMSetRenderTargets(1u, targetView.GetAddressOf(), nullptr);
	deviceContext->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);

	D3D11_VIEWPORT viewPort = {};
	viewPort.Width = 1600.0f;
	viewPort.Height = 800.0f;
	viewPort.MinDepth = 0.0f;
	viewPort.MaxDepth = 1.0f;
	viewPort.TopLeftX = 0.0f;
	viewPort.TopLeftY = 0.0f;
	deviceContext->RSSetViewports(1u, &viewPort);


	//camera.pos = Vec3f(0.0f, 0.0f, -2.0f);
	//camera.Z90 = Vec3f(0.0f, 0.0f, 1.0f);
	//camera.Y90 = Vec3f(0.0f, 1.0f, 0.0f);

	//auto camV = dx::XMVectorSet(camera.pos.x, camera.pos.y, camera.pos.z, 0.0f);
	//auto dir = dx::XMVectorSet(camera.Z90.x, camera.Z90.y, camera.Z90.z, 0.0f);
	//auto dirUp = dx::XMVectorSet(camera.Y90.x, camera.Y90.y, camera.Y90.z, 0.0f);

	//auto dirUp = dx::XMVectorSet(0.0f, 1.0f, 0.0f, 0.0f);

	struct cBuffer 
	{
		XMVECTOR camPos;
		XMMATRIX matrix;
	};

	//auto look = dx::XMVector3Transform(
	//	dx::XMVectorSet(0.0f, 0.0f, 1.0f, 0.0f),
	//	dx::XMMatrixRotationRollPitchYaw(pitch, yaw, 0.0f));

	//auto projection = ;
	//dx::XMVector3Project(, viewPort.TopLeftX, viewPort.TopLeftY, viewPort.Width, viewPort.Height, viewPort.MinDepth, viewPort.MaxDepth, );

	//auto camToLook = XMVectorAdd(camV, dir);
	cBuffer buff;
	//buff.camPos = camV;
	buff.camPos = camera.pos;
	buff.matrix = 
		XMMatrixIdentity() *
		view *
		XMMatrixPerspectiveFovLH(XM_2PI / 4.0f, viewPort.Width / viewPort.Height, 0.01f, 10000.0f);
	buff.matrix = XMMatrixTranspose(buff.matrix);
		
	ConstBufferVSSet(sizeof(cBuffer), &buff);

	deviceContext->Draw((UINT)verts.size(), 0u);
}

void DirectXGraphics::Clear(float r, float g, float b)
{
	float color[] = { r, g, b, 1.0f };
	deviceContext->ClearRenderTargetView(targetView.Get(), color);
}

void DirectXGraphics::CreateSetVShaderAndIL(LPCWCHAR fileName)
{
	Microsoft::WRL::ComPtr<ID3D11VertexShader> vertShader;
	Microsoft::WRL::ComPtr<ID3DBlob> blob;
	DXAssert(D3DReadFileToBlob(fileName, &blob));
	DXAssert(device->CreateVertexShader(blob->GetBufferPointer(), blob->GetBufferSize(), nullptr, &vertShader));
	deviceContext->VSSetShader(vertShader.Get(), nullptr, 0u);

	const D3D11_INPUT_ELEMENT_DESC elDesc[] =
	{
		{ "Position", 0u, DXGI_FORMAT_R32G32B32_FLOAT, 0u, 0u, D3D11_INPUT_PER_VERTEX_DATA, 0u },
	};
	Microsoft::WRL::ComPtr<ID3D11InputLayout> inputLayout;
	DXAssert(device->CreateInputLayout(elDesc, std::size(elDesc), blob->GetBufferPointer(), blob->GetBufferSize(), &inputLayout));
	deviceContext->IASetInputLayout(inputLayout.Get());
}

void DirectXGraphics::CreateSetPShader(LPCWCHAR fileName)
{
	Microsoft::WRL::ComPtr<ID3D11PixelShader> vertShader;
	Microsoft::WRL::ComPtr<ID3DBlob> blob;
	DXAssert(D3DReadFileToBlob(fileName, &blob));
	DXAssert(device->CreatePixelShader(blob->GetBufferPointer(), blob->GetBufferSize(), nullptr, &vertShader));
	deviceContext->PSSetShader(vertShader.Get(), nullptr, 0u);
}

void DirectXGraphics::ConstBufferVSSet(size_t bufferSize, void* dataFrom)
{
	D3D11_BUFFER_DESC desc = { };
	desc.ByteWidth = bufferSize;
	desc.Usage = D3D11_USAGE_DYNAMIC;
	desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
	desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
	desc.MiscFlags = 0u;
	desc.StructureByteStride = bufferSize;
	D3D11_SUBRESOURCE_DATA data = { };
	data.pSysMem = dataFrom;
	Microsoft::WRL::ComPtr<ID3D11Buffer> vertBuffer;
	DXAssert(device->CreateBuffer(&desc, &data, &vertBuffer));
	deviceContext->VSSetConstantBuffers(0u, 1u, vertBuffer.GetAddressOf());
}


