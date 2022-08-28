#include <DirectXMath.h>

using namespace DirectX;

class DXCamera
{
public:
	XMVECTOR defaultDir;
	XMVECTOR defaultUp;
	XMVECTOR defaultRight;
	XMVECTOR pos;
	XMVECTOR direction;
	XMVECTOR up;
	XMVECTOR right;
	float pitch;
	float yaw;
	XMMATRIX view;

public:
	DXCamera()
	{
		defaultDir = XMVectorSet(0, 0, 1, 0);
		defaultUp = XMVectorSet(0, 1, 0, 0);
		defaultRight = XMVectorSet(1, 0, 0, 0);
		pos = XMVectorSet(0, 0, -2, 0);
		direction = XMVectorSet(0, 0, 1, 0);
		up = XMVectorSet(0, 1, 0, 0);
		right = XMVectorSet(1, 0, 0, 0);
	}

	void Rotate(float pitch, float yaw)
	{
		this->pitch = pitch;
		this->yaw = yaw;
		auto rotation = XMMatrixRotationRollPitchYaw(pitch, yaw, 0.0f);
		direction = XMVector3TransformCoord(defaultDir, rotation);
		//up = XMVector3TransformCoord(defaultUp, rotation);
		right = XMVector3TransformCoord(defaultRight, rotation);
		//auto look = pos + direction;
		view = XMMatrixLookToLH(pos, direction, up);
	}
		
};

