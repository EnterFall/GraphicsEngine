#pragma once
#include <algorithm>

template <typename T>
struct Vec3
{
	T x;
	T y;
	T z;

	Vec3() = default;

	Vec3(T x1, T y1, T z1)
	{
		this->x = x1;
		this->y = y1;
		this->z = z1;
	}

	Vec3 operator +(const Vec3& other) const
	{
		return Vec3(x + other.x, y + other.y, z + other.z);
	}

	Vec3 operator -(const Vec3& other) const
	{
		return Vec3(x - other.x, y - other.y, z - other.z);
	}

	Vec3 operator *(const Vec3& other) const
	{
		return Vec3(x * other.x, y * other.y, z * other.z);
	}

	Vec3 operator /(const Vec3& other) const
	{
		return Vec3(x / other.x, y / other.y, z / other.z);
	}

	Vec3 operator *(const T& mul) const
	{
		return Vec3(x * mul, y * mul, z * mul);
	}

	Vec3 operator /(const T& mul) const
	{
		return Vec3(x / mul, y / mul, z / mul);
	}


	Vec3 Cross(const Vec3& other) const
	{
		return Vec3(
			y * other.z - z * other.y, 
			z * other.x - x * other.z, 
			x * other.y - y * other.x);
	}

	T Dot(const Vec3& other) const
	{
		return x * other.x + y * other.y + z * other.z;
	}

	T Length() const
	{
		return sqrt(x * x + y * y + z * z);
	}

	T LengthSquare() const
	{
		return x * x + y * y + z * z;
	}

	T Cos(const Vec3& other) const
	{
		return Dot(other) / Length() / other.Length();
	}

	void Normalize()
	{
		auto len = Length();
		x /= len;
		y /= len;
		z /= len;
	}
};
typedef Vec3<float> Vec3f;
typedef Vec3<double> Vec3d;
