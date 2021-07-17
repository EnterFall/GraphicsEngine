#pragma once
#include "Color.h"
#include "EFWin.h"
#include "EFException.h"
#include <algorithm>
#include <memory>
#include <string>

namespace Gdiplus
{
	using std::min;
	using std::max;
}
#include <gdiplus.h>
#pragma comment(lib,"gdiplus.lib")

class Texture
{
private:
	unsigned int width;
	unsigned int height;
	std::unique_ptr<Color[]> data;
public:
	Texture(const std::wstring& fileName);
	Color* GetData();
};

