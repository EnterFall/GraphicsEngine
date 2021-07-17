#pragma once
#include "Vec3.h"

template <typename T>
struct Matrix3
{
	T p00;
	T p01;
	T p02;
	T p10;
	T p11;
	T p12;
	T p20;
	T p21;
	T p22;

	Matrix3() :
		p00(1.0f), p01(0.0f), p02(0.0f),
		p10(0.0f), p11(1.0f), p12(0.0f),
		p20(0.0f), p21(0.0f), p22(1.0f)
	{

	}

	Matrix3(T x00, T x01, T x02, T x10, T x11, T x12, T x20, T x21, T x22) :
		p00(x00), p10(x10), p20(x20),
		p01(x01), p11(x11), p21(x21),
		p02(x02), p12(x12), p22(x22)
	{

	}

	Matrix3(const Vec3<T>& x0, const Vec3<T>& x1, const Vec3<T>& x2) :
		p00(x0.x), p10(x0.y), p20(x0.z),
		p01(x1.x), p11(x1.y), p21(x1.z),
		p02(x2.x), p12(x2.y), p22(x2.z)
	{
		
	}

	Matrix3& operator *=(const T& val)
	{
		p00 *= val;
		p01 *= val;
		p02 *= val;
		p10 *= val;
		p11 *= val;
		p12 *= val;
		p20 *= val;
		p21 *= val;
		p22 *= val;
		return *this;
	}

	Matrix3 Mult(const Matrix3& other) const
	{
		return Matrix3(
			p00 * other.p00 + p01 * other.p10 + p02 * other.p20,
			p00 * other.p01 + p01 * other.p11 + p02 * other.p21,
			p00 * other.p02 + p01 * other.p12 + p02 * other.p22,
			p10 * other.p00 + p11 * other.p10 + p12 * other.p20,
			p10 * other.p01 + p11 * other.p11 + p12 * other.p21,
			p10 * other.p02 + p11 * other.p12 + p12 * other.p22,
			p20 * other.p00 + p21 * other.p10 + p22 * other.p20,
			p20 * other.p01 + p21 * other.p11 + p22 * other.p21,
			p20 * other.p02 + p21 * other.p12 + p22 * other.p22);
	}

	Matrix3 Transpose() const
	{
		return Matrix3(
			p00, p10, p20,
			p01, p11, p21,
			p02, p12, p22);
	}

	Matrix3 Inverse() const
	{
		Vec3<T> x0 = Vec3<T>(p00, p10, p20);
		Vec3<T> x1 = Vec3<T>(p01, p11, p21);
		Vec3<T> x2 = Vec3<T>(p02, p12, p22);
		Vec3<T> x1CrossX2 = x1.Cross(x2);
		T det = x0.Dot(x1CrossX2);
		return Matrix3(x1CrossX2 / det, x2.Cross(x0) / det, x0.Cross(x1) / det).Transpose();
	}

	Matrix3 Inverse2() const
	{
		T a = p11 * p22 - p12 * p21;
		T b = p12 * p20 - p10 * p22;
		T c = p10 * p21 - p11 * p20;
		T det = p00 * a + p01 * b + p02 * c;
		return Matrix3(
			a, p02 * p21 - p01 * p22, p01 * p12 - p02 * p11,
			b, p00 * p22 - p02 * p20, p02 * p10 - p00 * p12,
			c, p01 * p20 - p00 * p21, p00 * p11 - p01 * p10) *= (1.0 / det);
	}

	Vec3<T> Mult(const Vec3<T>& vector) const
	{
		return Vec3(
			vector.x * p00 + vector.y * p01 + vector.z * p02,
			vector.x * p10 + vector.y * p11 + vector.z * p12,
			vector.x * p20 + vector.y * p21 + vector.z * p22);
	}
};
typedef Matrix3<double> Matrix3f;
typedef Matrix3<double> Matrix3d;