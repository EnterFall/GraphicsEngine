#pragma once
#include "CpuGraphics.h"
#include "Keyboard.h"
#include "RayMarchingGraphics.h"
#include "Window.h"
#include <chrono>


class SphereArrayScene_Device
{
private:
	RayMarchingGraphics* rayG;
	CpuGraphics* g;
	cudaSurfaceObject_t screenBuffer;
	cudaArray_t screenBuffer_array;
	Keyboard* kbd;
	Camera* cameraCuda;
	bool isFreeCamera;
	unsigned int size;
	float sphereRadius = 0.10f;
public:
	double time;
	SphereArrayScene_Device(CpuGraphics* g, Keyboard* kbd, unsigned int size) : g(g), kbd(kbd), size(size)
	{
		isFreeCamera = false;
		g->camera.pos = Vec3f(1.0f, 1.0f, 1.0f);
		g->camera.Rotate(MathHelper::PI_4 + 3.1f, MathHelper::PI_4 - 1.5f);
		
		cameraCuda = CudaHelper::CreateManagedCudaObj<Camera>();

		const int height = g->bufferHeight;
		const int width = g->bufferWidth;

		// Allocate CUDA arrays in device memory
		cudaChannelFormatDesc channelDesc =
			cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
		CudaAssert(cudaMallocArray(&screenBuffer_array, &channelDesc, width, height,
			cudaArraySurfaceLoadStore));
		
		// Specify surface
		struct cudaResourceDesc resDesc;
		memset(&resDesc, 0, sizeof(resDesc));
		resDesc.resType = cudaResourceTypeArray;
		resDesc.res.array.array = screenBuffer_array;
		CudaAssert(cudaCreateSurfaceObject(&screenBuffer, &resDesc));
		rayG = RayMarchingGraphics::Create(g->bufferWidth, g->bufferHeight, screenBuffer);
	}

	void Play()
	{
		float speed = g->travelSpeed / 10.0f;
		if (kbd->IsPressed('R'))
		{
			speed /= 10.0f;
		}
		if (kbd->IsPressed('J'))
		{
			speed /= 100.0f;
		}
		if (kbd->IsPressed('K'))
		{
			speed /= 1000.0f;
		}
		if (kbd->IsPressed('L'))
		{
			speed /= 10000.0f;
		}
		if (kbd->IsPressed('C'))
		{
			isFreeCamera = !isFreeCamera;
			Sleep(80);
		}
		if (kbd->IsPressed('B'))
		{
#ifdef _DEBUG
			DebugBreak();
#endif
		}
		if (kbd->IsPressed('W'))
		{
			g->camera.pos = g->camera.pos + (g->camera.Z90 * speed);
		}
		if (kbd->IsPressed('S'))
		{
			g->camera.pos = g->camera.pos - (g->camera.Z90 * speed);
		}
		if (kbd->IsPressed('D'))
		{
			g->camera.pos = g->camera.pos + (g->camera.X90 * speed);
		}
		if (kbd->IsPressed('A'))
		{
			g->camera.pos = g->camera.pos - (g->camera.X90 * speed);
		}
		if (kbd->IsPressed(VK_SPACE))
		{
			g->camera.pos = g->camera.pos + (g->camera.Y90 * speed);
		}
		if (kbd->IsPressed(VK_SHIFT))
		{
			g->camera.pos = g->camera.pos - (g->camera.Y90 * speed);
		}
		if (kbd->IsPressed('Y'))
		{
			sphereRadius += 0.001;
		}
		if (kbd->IsPressed('U'))
		{
			sphereRadius -= 0.001;
		}

		POINT e;
		if (!GetCursorPos(&e))
			throw WindowExceptLastError();

		if (isFreeCamera)
		{
			float x = (e.x - g->widthHalf) / (float)g->bufferHeight * MathHelper::PI2_f;
			float y = -(e.y - g->heightHalf) / (float)g->bufferHeight * MathHelper::PI2_f;

			g->camera.Rotate(x, y);
		}
		CudaAssert(cudaMemcpyAsync(cameraCuda, &g->camera, sizeof(Camera), cudaMemcpyHostToDevice));

		time = 0.0f;

		auto start = std::chrono::high_resolution_clock::now();
		

		

		CudaAssert(cudaMemcpy2DFromArrayAsync(g->screenBuffer.get(), g->bufferWidth * sizeof(Color), screenBuffer_array, 0, 0, g->bufferWidth * sizeof(Color), g->bufferHeight, cudaMemcpyDeviceToHost));
		CudaAssert(cudaDeviceSynchronize());
		rayG->Draw(cameraCuda, g->bufferWidth, g->bufferHeight, sphereRadius);
		
		time = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count();
	}

	void Dispose()
	{
		CudaAssert(cudaDeviceSynchronize());
		CudaAssert(cudaFree(rayG));
		CudaAssert(cudaFree(cameraCuda));
		CudaAssert(cudaDestroySurfaceObject(screenBuffer));
		CudaAssert(cudaFreeArray(screenBuffer_array));
	}
};