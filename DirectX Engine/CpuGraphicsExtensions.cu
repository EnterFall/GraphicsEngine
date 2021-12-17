#include "CpuGraphics.h"

void CpuGraphics::DrawTriangleFromToCuda(const Vec3f& leftS, const Vec3f& leftE, const Vec3f& rightS, const Vec3f& rightE, Color color)
{

	Vec3f vLeft = leftE - leftS;
	Vec3f vRight = rightE - rightS;

	float dxLeft = vLeft.x / vLeft.y;
	float dxRight = vRight.x / vRight.y;

	float dzLeft = vLeft.z / vLeft.y;

	float outLeft = clamp(-leftS.y / vLeft.y, 0.0f, 1.0f);
	float outRight = clamp(-rightS.y / vRight.y, 0.0f, 1.0f);

	int yStart = clamp((int)ceil(leftS.y - 0.5f), 0, bufferHeight);
	int yEnd = clamp((int)ceil(rightE.y - 0.5f), 0, bufferHeight);

	// if y < 0, start vectors will be corrected
	Vec3f leftCorr = leftS + vLeft * outLeft;
	Vec3f rightCorr = rightS + vRight * outRight;

	float t;
	float xLeft = leftCorr.x + dxLeft * modf(0.5f - leftCorr.y + float(yStart), &t);
	float xRight = rightCorr.x + dxRight * modf(0.5f - rightCorr.y + float(yStart), &t);

	float zLeft = leftCorr.z + dzLeft * modf(0.5f - leftCorr.y + float(yStart), &t);

	Vec3f d = ((rightE + rightS) * 0.5f) - ((leftE + leftS) * 0.5f);
	float dz = d.z / d.x;

	cudaStream_t stream;
	CudaAssert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
	int yCount = yEnd - yStart;
	if (yCount != 0)
	{
		RasterizeGlobal<<<(yCount / 32) + 1, 32, 0, stream>>>(gCudaPtr, yStart, yEnd, xLeft, xRight, zLeft, dxLeft, dxRight, dzLeft, dz, bufferWidth, color);
	}
	CudaAssert(cudaStreamDestroy(stream));
}