#include "EFException.h"

EFException::EFException(int line, std::string file) : line(line), file(file)
{}

EFException::EFException(int line, std::string file, const char* message) : line(line), file(file), message(message)
{}

int EFException::GetLine() const
{
	return line;
}

std::string EFException::GetFile() const
{
	return file;
}

std::string EFException::GetString() const
{
	std::ostringstream stream;
	stream << "[File] " << GetFile() << std::endl
		   << "[Line] " << GetLine()
		   << "[Mess] " << message;
	return stream.str();
}

const char* EFException::what() const
{
	std::ostringstream stream;
	stream << GetType() << std::endl
		   << GetString();
	whatBuffer = stream.str();
	return whatBuffer.c_str();
}

std::string EFException::GetType() const
{
	return "EF Exception";
}
