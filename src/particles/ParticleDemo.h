#ifndef INCLUDED_PARTICLEDEMO
#define INCLUDED_PARTICLEDEMO

#pragma once

#include <filesystem>


class ParticleDemo
{
public:
	void run(std::filesystem::path output_file, const std::filesystem::path& input_file, int N, int subsample, float dt, int cuda_device);
};

#endif  // INCLUDED_PARTICLEDEMO
