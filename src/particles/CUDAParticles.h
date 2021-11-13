#ifndef INCLUDED_CUDA_PARTICLES
#define INCLUDED_CUDA_PARTICLES

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <iosfwd>

#include <cuda_runtime_api.h>

#include <utils/CUDA/memory.h>
#include <utils/CUDA/event.h>

#include "particle_system_module.h"

#include "ParticleSystemLoader.h"


class ParticleReplayWriter;

class CUDAParticles
{
	particle_system_instance particles;

	std::size_t num_particles;

	CUDA::unique_ptr<float> position_buffer;
	CUDA::unique_ptr<std::uint32_t> color_buffer;

	CUDA::unique_event particles_begin;
	CUDA::unique_event particles_end;

	std::unique_ptr<float[]> position_download_buffer;
	std::unique_ptr<std::uint32_t[]> color_download_buffer;

public:
	CUDAParticles(particle_system_module& module, std::size_t num_particles, std::unique_ptr<float[]> position, std::unique_ptr<std::uint32_t[]> color, const ParticleSystemParameters& params);

	float update(std::ostream& file, ParticleReplayWriter& writer, int steps, float dt);
};

#endif // INCLUDED_CUDA_PARTICLES
