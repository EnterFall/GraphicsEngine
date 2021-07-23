#pragma once
#include "cuda_runtime.h"

struct Color
{
	unsigned int dword;

	__host__ __device__ Color() = default;

	__host__ __device__ Color(unsigned char r, unsigned char g, unsigned char b)
	{
		dword = (r << 16) | (g << 8) | b;
	} 

	__host__ __device__ Color(unsigned int val) : dword(val)
	{

	}

	__host__ __device__ unsigned int GetDWord()
	{
		return dword;
	}
};

//struct Colors
//{
//	static const constexpr Color White = Color(255u, 255u, 255u);
//	static const constexpr Color Red = Color(255u, 0u, 0u);
//	static const constexpr Color Green = Color(0u, 255u, 0u);
//	static const constexpr Color Blue = Color(0u, 0u, 255u);
//	static const constexpr Color Black = Color(0u, 0u, 0u);
//};