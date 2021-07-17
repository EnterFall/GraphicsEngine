#include "Texture.h"

Texture::Texture(const std::wstring& fileName)
{
	Gdiplus::Bitmap bitmap = Gdiplus::Bitmap(fileName.c_str());
	if (bitmap.GetLastStatus() != Gdiplus::Status::Ok)
	{
		throw EFException(__LINE__, __FILE__);
	}

	width = bitmap.GetWidth();
	height = bitmap.GetHeight();

	data = std::make_unique<Color[]>(height * width);
	for (unsigned int y = 0; y < height; y++)
	{
		for (unsigned int x = 0; x < width; x++)
		{
			Gdiplus::Color c;
			bitmap.GetPixel(x, y, &c);
			data[y * width + x] = c.GetValue();
		}
	}
}

Color* Texture::GetData()
{
	return data.get();
}
