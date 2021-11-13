#ifndef INCLUDED_PARTICLE_SYSTEM_STATE
#define INCLUDED_PARTICLE_SYSTEM_STATE

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <chrono>
#include <iosfwd>
#include <filesystem>

#include <utils/io/compression.h>

#include "particle_system_module.h"


struct ParticleReplayBuilder
{
	virtual void add_frame(std::chrono::nanoseconds dt, const float* positions, const std::uint32_t* colors) = 0;

protected:
	ParticleReplayBuilder() = default;
	ParticleReplayBuilder(const ParticleReplayBuilder&) = default;
	ParticleReplayBuilder(ParticleReplayBuilder&&) = default;
	ParticleReplayBuilder& operator =(const ParticleReplayBuilder&) = default;
	ParticleReplayBuilder& operator =(ParticleReplayBuilder&&) = default;
	~ParticleReplayBuilder() = default;
};

struct ParticleSystemBuilder
{
	virtual void add_particle_simulation(std::size_t num_particles, std::unique_ptr<float[]> position, std::unique_ptr<std::uint32_t[]> color, const ParticleSystemParameters& params) = 0;
	virtual ParticleReplayBuilder& add_particle_replay(std::size_t num_particles, const float* position, const std::uint32_t* color, const ParticleSystemParameters& params) = 0;

protected:
	ParticleSystemBuilder() = default;
	ParticleSystemBuilder(const ParticleSystemBuilder&) = default;
	ParticleSystemBuilder(ParticleSystemBuilder&&) = default;
	ParticleSystemBuilder& operator =(const ParticleSystemBuilder&) = default;
	ParticleSystemBuilder& operator =(ParticleSystemBuilder&&) = default;
	~ParticleSystemBuilder() = default;
};


class ParticleReplayWriter
{
	zlib_writer writer;
	std::size_t num_particles;

public:
	ParticleReplayWriter(std::ostream& file, std::size_t num_particles, const float* x, const float* y, const float* z, const float* r, const std::uint32_t* color, const ParticleSystemParameters& params);

	void add_frame(std::ostream& file, std::chrono::nanoseconds dt, const float* positions, const std::uint32_t* colors);

	std::ostream& finish(std::ostream& file);
};


std::ostream& save_particle_state(std::ostream& file, std::size_t num_particles, const float* x, const float* y, const float* z, const float* r, const std::uint32_t* color, const ParticleSystemParameters& params);
void save_particle_state(const std::filesystem::path& filename, std::size_t num_particles, const float* x, const float* y, const float* z, const float* r, const std::uint32_t* color, const ParticleSystemParameters& params);

std::istream& load_particles(ParticleSystemBuilder& builder, std::istream& file);
void load_particles(ParticleSystemBuilder& builder, const std::filesystem::path& filename);

#endif // INCLUDED_PARTICLE_SYSTEM_STATE
