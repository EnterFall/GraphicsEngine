#pragma once
#include "cuda_runtime.h"
#include <exception>
#include <sstream>
#include <string>

class EFException : public std::exception
{
private:
	int line;
	std::string file;
	const char* message;
protected:
	mutable std::string whatBuffer;
public:
	EFException(int line, std::string file);
	EFException(int line, std::string file, const char* message);
	int GetLine() const;
	std::string GetFile() const;
	std::string GetString() const;
	const char* what() const override;
	virtual std::string GetType() const;
};