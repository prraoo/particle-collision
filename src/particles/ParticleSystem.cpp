#include "ParticleSystem.h"
#include <iostream>
#include <math.h>

ParticleSystem::ParticleSystem(std::size_t num_particles, const float* x, const float* y, const float* z, const float* r, const std::uint32_t* color, const ParticleSystemParameters& params)
	: num_particles(num_particles)
	, params(params)
{
	void* p1; void* p2; void* p3;
	void* g1;
	void* f1; void* f2;

	cudaMalloc(&p1, num_particles * sizeof(float3));
	cudaMalloc(&p2, num_particles * 4 * sizeof(float));
	cudaMalloc(&p3, num_particles * sizeof(std::uint32_t));
	
	cudaMalloc(&g1, num_particles * sizeof(uint2));

	int resolution = powf((params.bb_max[0] - params.bb_min[0]) / (params.max_particle_radius * 2),3); // side x side x side 
	cudaMalloc(&f1, resolution * sizeof(int));
	cudaMalloc(&f2, resolution * sizeof(int));


	prev_pos = static_cast<float3*>(p1);
	pos = static_cast<float*>(p2);
	c = static_cast<uint32_t*>(p3);

	grid_hash_index = static_cast<uint2*>(g1);
	cell_begin_idx = static_cast<int*>(f1);
	cell_end_idx = static_cast<int*>(f2);
	

	reset(x, y, z, r, color);
}

void update_particles(float* position, 	float3* prev_pos, float dt, uint2* grid_hash_index, int* cell_begin_idx, int* cell_end_idx,
	ParticleSystemParameters params, std::size_t num_particles);

void init_particles(float* position, std::uint32_t* color, float3* prev_pos, float* pos, std::uint32_t* clr, ParticleSystemParameters params,
	std::size_t num_particles);

void ParticleSystem::reset(const float* x, const float* y, const float* z, const float* r, const std::uint32_t* color)
{
	sim_steps = 0;
	
	cudaMemcpy(pos + 0 * num_particles, x, num_particles * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(pos + 1 * num_particles, y, num_particles * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(pos + 2 * num_particles, z, num_particles * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(pos + 3 * num_particles, r, num_particles * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(c, color, num_particles * sizeof(std::uint32_t), cudaMemcpyHostToDevice);
}

void ParticleSystem::update(float* position, std::uint32_t* color, float dt)
{
	if (sim_steps == 0) {
		init_particles(position, color,  prev_pos, pos, c, params, num_particles);	
	}

	update_particles(position, prev_pos, dt, grid_hash_index, cell_begin_idx, cell_end_idx, params, num_particles);
	
	cudaFree(grid_hash_index);
	sim_steps++;
}

