#include <cstdint>
#include <cmath>
#include <iostream>
#include <iomanip>

#include <cuda_runtime_api.h>

#include <utils/image.h>
#include <utils/io/png.h>

#include <utils/math/vector.h>

#include <utils/CUDA/error.h>
#include <utils/CUDA/memory.h>
#include <utils/CUDA/array.h>
#include <utils/CUDA/event.h>

#include "envmap.h"
#include "HDRPipeline.h"
#include "HDRDemo.h"


namespace
{
	std::ostream& pad(std::ostream& out, int n)
	{
		for (int i = n; i > 0; --i) out.put(' ');
		return out;
	}
}

void HDRDemo::run(const std::filesystem::path& output_file, const std::filesystem::path& envmap_path, float exposure, float brightpass_threshold, int test_runs)
{
	if (!vertices.empty() || !indices.empty())
		std::cerr << "\nWARNING: scene geometry ignored in noninteractive mode\n";

	std::cout << "\nreading " << envmap_path << '\n' << std::flush;

	auto envmap = load_envmap(envmap_path, false);

	int image_width = static_cast<int>(width(envmap));
	int image_height = static_cast<int>(height(envmap));

	auto hdr_frame = CUDA::create_array(width(envmap), height(envmap), { 32, 32, 32, 32, cudaChannelFormatKindFloat });
	auto ldr_frame = CUDA::create_array(width(envmap), height(envmap), { 8, 8, 8, 8, cudaChannelFormatKindUnsigned });

	throw_error(cudaMemcpy2DToArray(hdr_frame.get(), 0, 0, data(envmap), image_width * 16U, image_width * 16U, image_height, cudaMemcpyHostToDevice));

	HDRPipeline pipeline(image_width, image_height);


	auto pipeline_begin = CUDA::create_event();
	auto pipeline_end = CUDA::create_event();

	float pipeline_time = 0.0f;

	std::cout << '\n' << test_runs << " test run(s):\n";

	int padding = static_cast<int>(std::log10(test_runs));
	int next_padding_shift = 10;

	for (int i = 0; i < test_runs; ++i)
	{
		throw_error(cudaEventRecord(pipeline_begin.get()));
		pipeline.process(ldr_frame.get(), hdr_frame.get(), exposure, brightpass_threshold);
		throw_error(cudaEventRecord(pipeline_end.get()));

		throw_error(cudaEventSynchronize(pipeline_end.get()));

		auto t = CUDA::elapsed_time(pipeline_begin.get(), pipeline_end.get());


		if ((i + 1) >= next_padding_shift)
		{
			--padding;
			next_padding_shift *= 10;
		}

		pad(std::cout, padding) << "t_" << (i + 1) << ": " << std::setprecision(2) << std::fixed << t << " ms\n" << std::flush;

		pipeline_time += t;
	}

	std::cout << "avg time: " << std::setprecision(2) << std::fixed << pipeline_time / test_runs << " ms\n" << std::flush;


	image2D<std::uint32_t> output(image_width, image_height);
	throw_error(cudaMemcpy2DFromArray(data(output), width(output) * 4U, ldr_frame.get(), 0, 0, image_width * 4U, image_height, cudaMemcpyDeviceToHost));

	std::cout << "\nsaving " << output_file << '\n' << std::flush;
	PNG::saveImageR8G8B8(output_file.string().c_str(), output);
}
