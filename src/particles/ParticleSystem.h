#ifndef INCLUDED_PARTICLE_SYSTEM
#define INCLUDED_PARTICLE_SYSTEM
#pragma once

#include <cuda_runtime_api.h>
#include "particle_system_module.h"


class ParticleSystem
{
	const std::size_t num_particles;
	const ParticleSystemParameters params;

	float* pos;
	float3* prev_pos;
	std::uint32_t* c;
	
	uint2* grid_hash_index;
	int* cell_begin_idx;
	int* cell_end_idx;
	unsigned int sim_steps = 0;

public:
	ParticleSystem(std::size_t num_particles, const float* x, const float* y, const float* z, const float* r, const std::uint32_t* color, const ParticleSystemParameters& params);

	void reset(const float* x, const float* y, const float* z, const float* r, const std::uint32_t* color);
	void update(float* position, std::uint32_t* color, float dt);
};

#endif // INCLUDED_PARTICLE_SIMULATION
