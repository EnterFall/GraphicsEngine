#pragma once
#include <algorithm>
#include "cuda_runtime.h"

template <typename T>
struct Vec3
{
	T x;
	T y;
	T z;

	__host__ __device__ Vec3() = default;

	__host__ __device__ Vec3(T x, T y, T z)
	{
		this->x = x;
		this->y = y;
		this->z = z;
	}

	__host__ __device__ Vec3 operator +(const Vec3& other) const
	{
		return Vec3(x + other.x, y + other.y, z + other.z);
	}

	__host__ __device__ Vec3 operator -(const Vec3& other) const
	{
		return Vec3(x - other.x, y - other.y, z - other.z);
	}

	__host__ __device__ Vec3 operator *(const Vec3& other) const
	{
		return Vec3(x * other.x, y * other.y, z * other.z);
	}

	__host__ __device__ Vec3 operator /(const Vec3& other) const
	{
		return Vec3(x / other.x, y / other.y, z / other.z);
	}

	__host__ __device__ Vec3& operator +=(const Vec3& other)
	{
		x += other.x;
		y += other.y;
		z += other.z;
		return *this;
	}

	__host__ __device__ Vec3 operator +(const T& val) const
	{
		return Vec3(x + val, y + val, z + val);
	}

	__host__ __device__ Vec3 operator -(const T& val) const
	{
		return Vec3(x - val, y - val, z - val);
	}

	__host__ __device__ Vec3 operator *(const T& val) const
	{
		return Vec3(x * val, y * val, z * val);
	}

	__host__ __device__ Vec3 operator /(const T& val) const
	{
		T v = 1.0f / val;
		return Vec3(x * v, y * v, z * v);
	}

	__host__ __device__ Vec3& operator +=(const T& val)
	{
		x += val;
		y += val;
		z += val;
		return *this;
	}

	__host__ __device__ bool operator ==(const Vec3& other) const
	{
		return x == other.x && y == other.y && z == other.z;
	}

	__host__ __device__ bool operator !=(const Vec3& other) const
	{
		return x != other.x || y != other.y || z != other.z;
	}

	__host__ __device__ Vec3 Cross(const Vec3& other) const
	{
		return Vec3(
			y * other.z - z * other.y, 
			z * other.x - x * other.z, 
			x * other.y - y * other.x);
	}

	__host__ __device__ T Dot(const Vec3& other) const
	{
		return x * other.x + y * other.y + z * other.z;
	}

	__host__ __device__ T Length() const
	{
		return sqrt(x * x + y * y + z * z);
	}

	__host__ __device__ T LengthSquare() const
	{
		return x * x + y * y + z * z;
	}

	__host__ __device__ T Cos(const Vec3& other) const
	{
		return Dot(other) / (Length() * other.Length());
	}

	__host__ __device__ void Normalize()
	{
		auto len = 1.0f / Length();
		x *= len;
		y *= len;
		z *= len;
	}
};
typedef Vec3<int> Vec3i;
typedef Vec3<float> Vec3f;
typedef Vec3<double> Vec3d;
