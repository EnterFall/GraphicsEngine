#include "CubesArrayScene_CUDA.h"

CubesArrayScene_CUDA::CubesArrayScene_CUDA(CpuGraphics* g, Keyboard* kbd, unsigned int size) : g(g), kbd(kbd), size(size)
{
	srand(3454);
	g->camera.pos = Vec3f(-50.0f, 50.0f, -50.0f) / 5;
	g->camera.Rotate(MathHelper::PI_4, -MathHelper::PI_4 + 0.2f);

	gCuda = CudaHelper::CreateManagedCudaObj<GpuGraphics>();
	screenBufferCuda = CudaHelper::CreateManagedCudaObj<Color>(g->bufferWidth * g->bufferHeight);
	cameraCuda = CudaHelper::CreateManagedCudaObj<Camera>();
	ConstructGpuGraphics<<<1, 1>>>(gCuda, screenBufferCuda, cameraCuda);
	CudaAssert(cudaGetLastError());
	CudaAssert(cudaDeviceSynchronize());
	cubes = std::vector<Cube>();
	for (unsigned int z = 0; z < size; z++)
	{
		// y from far to near, so ZBuffer almost dont reject drawing. Even if y goes to far, fps is not improved.
		for (int y = -(int)size; y < 0; y++)
		{
			for (unsigned int x = 0; x < size; x++)
			{
				if (rand() % 4 == 0)
				{
					cubes.emplace_back(Vec3f(x, y, z), 1.0f);
				}
			}
		}
	}
	cubesCuda = CudaHelper::CreateCopyToManagedCudaObj<Cube>(cubes.begin()._Ptr, cubes.size());
}

void CubesArrayScene_CUDA::Play()
{
	float speed = g->travelSpeed;
	if (kbd->IsPressed('R'))
	{
		speed /= 10.0f;
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
	if (kbd->IsPressed('T'))
	{
		g->isMatrixTransform = !g->isMatrixTransform;
		Sleep(80);
	}

	POINT e;
	if (!GetCursorPos(&e))
	{

	}

	if (isFreeCamera)
	{
		float x = (e.x - g->widthHalf) / (float)g->bufferHeight * MathHelper::PI2_f;
		float y = -(e.y - g->heightHalf) / (float)g->bufferHeight * MathHelper::PI2_f;

		g->camera.Rotate(x, y);
	}

	dim3 threadsPerBlock = dim3(8, 8);
	dim3 numBlocks = dim3(g->bufferWidth / threadsPerBlock.x, g->bufferHeight / threadsPerBlock.y);
	Clear<<<numBlocks, threadsPerBlock>>>(gCuda);
	CudaAssert(cudaGetLastError());
	CudaAssert(cudaDeviceSynchronize());
	CudaAssert(cudaMemcpyAsync(cameraCuda, &g->camera, sizeof(Camera), cudaMemcpyHostToDevice));

	//auto tr = std::vector<Vec3f>();
	//g->tr = &tr;

	//g->gCudaPtr = gCuda;
	//for (int i = 0; i < cubes.size(); i++)
	//{
	//	g->Draw(cubes[i], 0xFFFFFF);
	//}
	
	//auto trCuda = CudaHelper::CreateCopyToManagedCudaObj<Vec3f>(tr.begin()._Ptr, tr.size());
	//DrawTriangleFromTo<<<((tr.size() / 4) / 32) + 1, 32>>>(gCuda, trCuda, tr.size() / 4, 0xFFFFFF);
	//CudaAssert(cudaFree(trCuda));
	// 
	DrawCubes<<<((cubes.size() * 12) / 32) + 1, 32>>>(gCuda, cubesCuda, cubes.size(), size);

	CudaAssert(cudaGetLastError());
	CudaAssert(cudaDeviceSynchronize());
	CudaAssert(cudaMemcpyAsync(g->screenBuffer.get(), screenBufferCuda, g->bufferWidth * g->bufferHeight * sizeof(Color), cudaMemcpyDeviceToHost));

}