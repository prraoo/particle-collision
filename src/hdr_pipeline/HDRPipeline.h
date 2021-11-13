#ifndef INCLUDED_HDRPIPELINE
#define INCLUDED_HDRPIPELINE

#pragma once

#include <cstdint>

#include <cuda_runtime_api.h>

#include <utils/CUDA/memory.h>


class HDRPipeline
{
	int frame_width;
	int frame_height;

	CUDA::unique_ptr<float> d_input_image;
	CUDA::unique_ptr<std::uint32_t> d_output_image;

public:
	HDRPipeline(int width, int height);

	void process(cudaArray_t out, cudaArray_t in, float exposure, float brightpass_threshold);
};

#endif // INCLUDED_HDRPIPELINE
