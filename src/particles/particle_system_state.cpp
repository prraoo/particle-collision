#include <cstdint>
#include <stdexcept>
#include <memory>
#include <fstream>
#include <string_view>
#include <iostream>

#include <zlib.h>

#include <utils/io/compression.h>
#include <utils/io.h>

#include "particle_system_state.h"

using namespace std::literals;


namespace
{
	template <typename T>
	auto alloc_buffer(std::size_t num_particles)
	{
		return std::unique_ptr<T[]>(new T[num_particles]);
	}

	std::ostream& write_header(std::ostream& file, std::size_t num_particles, bool is_replay)
	{
		if (!write<std::int64_t>(file, is_replay ? -static_cast<std::int64_t>(num_particles) : static_cast<std::int64_t>(num_particles)))
			throw std::runtime_error("failed to write particles file header");

		return file;
	}

	auto read_header(std::istream& file)
	{
		auto num_particles = read<std::int64_t>(file);

		if (!file)
			throw std::runtime_error("failed to read particles file header");


		struct header_info
		{
			std::size_t num_particles;
			bool is_replay;
		};

		return header_info { static_cast<std::size_t>(num_particles < 0 ? -num_particles : num_particles), num_particles < 0 };
	}

	void write(std::ostream& file, zlib_writer& writer, const ParticleSystemParameters& params)
	{
		writer(file, params.bb_min);
		writer(file, params.bb_max);
		writer(file, params.min_particle_radius);
		writer(file, params.max_particle_radius);
		writer(file, params.gravity);
		writer(file, params.bounce);
		writer(file, params.coll_attraction);
		writer(file, params.coll_damping);
		writer(file, params.coll_shear);
		writer(file, params.coll_spring);
	}

	void read(ParticleSystemParameters& params, zlib_reader& reader, std::istream& file)
	{
		reader(params.bb_min, file);
		reader(params.bb_max, file);
		reader(params.min_particle_radius, file);
		reader(params.max_particle_radius, file);
		reader(params.gravity, file);
		reader(params.bounce, file);
		reader(params.coll_attraction, file);
		reader(params.coll_damping, file);
		reader(params.coll_shear, file);
		reader(params.coll_spring, file);
	}

	void write_initial_particle_state(std::ostream& file, zlib_writer& writer, std::size_t num_particles, const float* x, const float* y, const float* z, const float* r, const std::uint32_t* color, const ParticleSystemParameters& params)
	{
		writer(file, params);
		writer(file, &x[0], num_particles);
		writer(file, &y[0], num_particles);
		writer(file, &z[0], num_particles);
		writer(file, &r[0], num_particles);
		writer(file, &color[0], num_particles);
	}

	void read_initial_particle_state(zlib_reader& reader, std::size_t num_particles, float* x, float* y, float* z, float* r, std::uint32_t* color, ParticleSystemParameters& params, std::istream& file)
	{
		reader(params, file);
		reader(&x[0], num_particles, file);
		reader(&y[0], num_particles, file);
		reader(&z[0], num_particles, file);
		reader(&r[0], num_particles, file);
		reader(&color[0], num_particles, file);
	}

	void write_frame_data(std::ostream& file, zlib_writer& writer, std::size_t num_particles, std::chrono::nanoseconds dt, const float* positions, const std::uint32_t* colors)
	{
		writer(file, static_cast<std::int64_t>(dt.count()));
		writer(file, positions, num_particles * 4);
		writer(file, colors, num_particles);
	}

	zlib_reader& read_frame_data(zlib_reader& reader, std::size_t num_particles, std::chrono::nanoseconds& dt, float* positions, std::uint32_t* colors, std::istream& file)
	{
		dt = std::chrono::nanoseconds(reader.read<std::int64_t>(file));
		reader(positions, num_particles * 4, file);
		reader(colors, num_particles, file);

		return reader;
	}
}

std::ostream& save_particle_state(std::ostream& file, std::size_t num_particles, const float* x, const float* y, const float* z, const float* r, const std::uint32_t* color, const ParticleSystemParameters& params)
{
	write_header(file, num_particles, false);

	zlib_writer writer;
	write_initial_particle_state(file, writer, num_particles, x, y, z, r, color, params);

	return writer.finish(file);
}

ParticleReplayWriter::ParticleReplayWriter(std::ostream& file, std::size_t num_particles, const float* x, const float* y, const float* z, const float* r, const std::uint32_t* color, const ParticleSystemParameters& params)
	: num_particles(num_particles)
{
	write_header(file, num_particles, true);
	write_initial_particle_state(file, writer, num_particles, x, y, z, r, color, params);
}

void ParticleReplayWriter::add_frame(std::ostream& file, std::chrono::nanoseconds dt, const float* positions, const std::uint32_t* colors)
{
	write_frame_data(file, writer, num_particles, dt, positions, colors);
}

std::ostream& ParticleReplayWriter::finish(std::ostream& file)
{
	return writer.finish(file);
}

std::istream& load_particles(ParticleSystemBuilder& builder, std::istream& file)
{
	auto [num_particles, is_replay] = read_header(file);

	zlib_reader reader;

	ParticleSystemParameters params;
	auto position = alloc_buffer<float>(num_particles * 4);
	auto color = alloc_buffer<std::uint32_t>(num_particles);

	read_initial_particle_state(reader, num_particles, &position[0] + 0 * num_particles, &position[0] + 1 * num_particles, &position[0] + 2 * num_particles, &position[0] + 3 * num_particles, &color[0], params, file);

	if (is_replay)
	{
		auto& replay_builder = builder.add_particle_replay(num_particles, &position[0], &color[0], params);

		while (reader)
		{
			std::chrono::nanoseconds dt;
			read_frame_data(reader, num_particles, dt, &position[0], &color[0], file);
			replay_builder.add_frame(dt, &position[0], &color[0]);
		}
	}
	else
	{
		builder.add_particle_simulation(num_particles, std::move(position), std::move(color), params);
	}

	return file;
}


void save_particle_state(const std::filesystem::path& filename, std::size_t num_particles, const float* x, const float* y, const float* z, const float* r, const std::uint32_t* colors, const ParticleSystemParameters& params)
{
	std::ofstream file(filename, std::ios::binary);

	if (!file)
		throw std::runtime_error("failed to open '" + filename.string() + '\'');

	save_particle_state(file, num_particles, x, y, z, r, colors, params);
}

void load_particles(ParticleSystemBuilder& builder, const std::filesystem::path& filename)
{
	std::ifstream file(filename, std::ios::binary);

	if (!file)
		throw std::runtime_error("failed to open '" + filename.string() + '\'');

	load_particles(builder, file);
}
