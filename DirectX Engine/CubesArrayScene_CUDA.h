#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "GpuGraphics.h"
#include "CudaHelper.h"
#include "Window.h"
#include "MathHelper.h"


class CubesArrayScene_CUDA
{
private:
	GpuGraphics* gCuda;
	Color* screenBufferCuda;
	CpuGraphics* g;
	Keyboard* kbd;
	std::vector<Cube> cubes;
	Cube* cubesCuda;
	Camera* cameraCuda;
	bool isFreeCamera;
	unsigned int size;
public:
	CubesArrayScene_CUDA(CpuGraphics* g, Keyboard* kbd, unsigned int size);
	void Play();
};