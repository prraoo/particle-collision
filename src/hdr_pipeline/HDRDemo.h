#ifndef INCLUDED_HDRDEMO
#define INCLUDED_HDRDEMO

#pragma once

#include <cstdint>
#include <vector>
#include <filesystem>

#include <utils/io/obj.h>


class HDRDemo : private virtual OBJ::MeshSink
{
	math::float3 bb_min = {  1.0f,  1.0f,  1.0f };
	math::float3 bb_max = { -1.0f, -1.0f, -1.0f };

	std::vector<float> vertices;
	std::vector<std::uint32_t> indices;

	int add_vertex(const math::float3& position, const math::float3& normal) override;
	void add_triangle(int v_1, int v_2, int v_3) override;

	void run(const std::filesystem::path& output_file, const std::filesystem::path& envmap, float exposure, float brightpass_threshold, int test_runs);

public:
	void add_model(const std::filesystem::path& path);

	void run(const std::filesystem::path& output_file, const std::filesystem::path& envmap, int cuda_device, float exposure_value, float brightpass_threshold, int test_runs);
};

#endif  // INCLUDED_HDRDEMO
