#include <cstdint>
#include <iostream>
#include "particle_system_module.h"
#include "vector_operations.cuh"

#include <utils/CUDA/error.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

__constant__ int d_MAX_GRIDSIZE;
__constant__ int STEP;


__device__ void reflect_particle(float3& pos_t, float3& pos_t_1, float radius, ParticleSystemParameters params) {
	float bb_threshold;
	
	if (pos_t_1.x  < (params.bb_min[0] + radius)) {
		bb_threshold = (params.bb_min[0] + radius);
		pos_t.x = bb_threshold + params.bounce * (bb_threshold - pos_t.x);
		pos_t_1.x = bb_threshold + params.bounce * (bb_threshold - pos_t_1.x);
	}
	else if (pos_t_1.x  > (params.bb_max[0] - radius)) {
		bb_threshold = (params.bb_max[0] - radius);
		pos_t.x = bb_threshold - params.bounce * (pos_t.x - bb_threshold);
		pos_t_1.x = bb_threshold - params.bounce * (pos_t_1.x - bb_threshold);
	}
	else if (pos_t_1.y < (params.bb_min[1] + radius)) {
		bb_threshold = (params.bb_min[1] + radius);
		pos_t.y = bb_threshold + params.bounce * (bb_threshold - pos_t.y);
		pos_t_1.y = bb_threshold + params.bounce * (bb_threshold - pos_t_1.y);
	}
	else if (pos_t_1.y > (params.bb_max[1] - radius)) {
		bb_threshold = (params.bb_max[1] - radius);
		pos_t.y = bb_threshold - params.bounce * (pos_t.y - bb_threshold);
		pos_t_1.y = bb_threshold - params.bounce * (pos_t_1.y - bb_threshold);
	}
	else if (pos_t_1.z < (params.bb_min[2] + radius)) {
		bb_threshold = (params.bb_min[2] + radius);
		pos_t.z = bb_threshold + params.bounce * (bb_threshold - pos_t.z);
		pos_t_1.z = bb_threshold + params.bounce * (bb_threshold - pos_t_1.z);
	}
	else if (pos_t_1.z > (params.bb_max[2] - radius)) {
		bb_threshold = (params.bb_max[2] - radius);
		pos_t.z = bb_threshold - params.bounce * (pos_t.z - bb_threshold);
		pos_t_1.z = bb_threshold - params.bounce * (pos_t_1.z - bb_threshold);
	}
}

// Check bounding box limits
__device__ bool check_outside(float3& pos, float radius, ParticleSystemParameters params) {
	return (pos.x < (params.bb_min[0] + radius) || pos.x  >(params.bb_max[0] - radius) ||
		pos.y < (params.bb_min[1] + radius) || pos.y  >(params.bb_max[1] - radius) ||
		pos.z < (params.bb_min[2] + radius) || pos.z  >(params.bb_max[2] - radius));
}

__device__ bool invalid_index(int3 particle_index) {
	return (particle_index.x < 0 || particle_index.x >= STEP ||
		particle_index.y < 0 || particle_index.y >= STEP ||
		particle_index.z < 0 || particle_index.z >= STEP);
}

struct sort_uint2 {
	__host__ __device__ bool operator()(const uint2& a, const uint2& b) const {
		//https://stackoverflow.com/questions/33027336/how-to-sort-an-array-of-cuda-vector-types
		return (a.x < b.x);
	}
};

__device__ float3 calculate_acceleration(
	float* position,
	float3* prev_pos,
	uint2* grid_hash_index,
	int* cell_begin_idx,
	int* cell_end_idx,
	int3 neighbour_index,
	/*unsigned int neighbour_hash,*/
	/*unsigned int start_id,*/
	float3 pos_t_a,
	float3 pos_t_1_a,
	float r_a,
	ParticleSystemParameters params,
	std::size_t num_particles,
	float dt) {

	float3 resultant_force = make_float3(0, 0, 0);

	
	if (invalid_index(neighbour_index) == true) {
		return resultant_force;
	}
	unsigned int neighbour_hash = floorf(neighbour_index.x + neighbour_index.y * STEP + neighbour_index.z * STEP * STEP);
	unsigned int start_idx = cell_begin_idx[neighbour_hash];

	if (start_idx < num_particles) {

		unsigned int end_idx = cell_end_idx[neighbour_hash];
		uint2 particle_b = grid_hash_index[start_idx];

		for (unsigned int idx=start_idx; idx<end_idx;idx++){

			particle_b = grid_hash_index[idx];

			float3 pt_b;
			pt_b.x = position[particle_b.y * 4 + 0];
			pt_b.y = position[particle_b.y * 4 + 1];
			pt_b.z = position[particle_b.y * 4 + 2];
			float r_b = position[particle_b.y * 4 + 3];

			float3 pos_t_ba = pt_b - pos_t_a;
			float dist_ba = sqrtf(pos_t_ba.x * pos_t_ba.x + pos_t_ba.y * pos_t_ba.y + pos_t_ba.z * pos_t_ba.z);

			//equation (1)
			if (dist_ba > 0 && dist_ba < (r_a + r_b) ) {
				float3 pos_t_1_b = prev_pos[particle_b.y];
				//equation (2)
				float3 v_a = (pos_t_a - pos_t_1_a) / dt;
				float3 v_b = (pt_b - pos_t_1_b) / dt;
				//equation (3)
				float3 dir_ba = (pos_t_ba) / dist_ba;
				float3 v_ba = v_b - v_a;
				//equation (4)
				float3 v_ab = v_ba - dir_ba * (v_ba.x * dir_ba.x + v_ba.y * dir_ba.y + v_ba.z * dir_ba.z);
				//equation (5)
				resultant_force += ((-params.coll_spring * (r_a + r_b - dist_ba)) * dir_ba);
				//equation (6)
				resultant_force += (params.coll_damping * v_ba);
				//equation (7)
				resultant_force += (params.coll_shear * v_ab);
			}
		}
	}

	return resultant_force;
}


// Initalize the position
__global__ void init_position_kernel(
	float* position, 
	std::uint32_t* color, 
	float3* prev_pos, 
	float* pos, 
	std::uint32_t* c, 
	ParticleSystemParameters params,
	std::size_t num_particles) {

	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < num_particles) {
		float3 pos_t;

		pos_t.x = pos[0 * num_particles + tid];
		pos_t.y = pos[1 * num_particles + tid];
		pos_t.z = pos[2 * num_particles + tid];
		float r = pos[3 * num_particles + tid];

		position[tid * 4 + 0] = pos_t.x;
		position[tid * 4 + 1] = pos_t.y;
		position[tid * 4 + 2] = pos_t.z;
		position[tid * 4 + 3] = r;

		prev_pos[tid] = pos_t;
		color[tid] = c[tid];
	}
}

void init_particles(
	float* position, 
	std::uint32_t* color, 
	float3* prev_pos, 
	float* pos, 
	std::uint32_t* c, 
	ParticleSystemParameters params,
	std::size_t num_particles) {

	int MAX_GRIDSIZE = 2 * params.max_particle_radius;
	int step = (params.bb_max[0] - params.bb_min[0]) / MAX_GRIDSIZE;

	cudaMemcpyToSymbol(d_MAX_GRIDSIZE, &MAX_GRIDSIZE, sizeof(int));
	cudaMemcpyToSymbol(STEP, &step, sizeof(int));

	constexpr int BLOCK_SIZE = 128;
	auto NUM_BLOCKS = (num_particles + BLOCK_SIZE - 1) / BLOCK_SIZE;

	init_position_kernel <<< NUM_BLOCKS, BLOCK_SIZE >>> (position, color, prev_pos, pos, c, params, num_particles);
}

__global__ void calculate_hash_kernel(
	float* position,
	uint2* grid_hash_index,
	ParticleSystemParameters params,
	std::size_t num_particles) {

	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < num_particles) {

		float3 pos_t;
		int3 particle_index;
		pos_t.x = position[tid * 4 + 0];
		pos_t.y = position[tid * 4 + 1];
		pos_t.z = position[tid * 4 + 2];

		particle_index.x = (pos_t.x - params.bb_min[0]) / d_MAX_GRIDSIZE;
		particle_index.y = (pos_t.y - params.bb_min[1]) / d_MAX_GRIDSIZE;
		particle_index.z = (pos_t.z - params.bb_min[2]) / d_MAX_GRIDSIZE;

		unsigned int particle_hash = floorf(particle_index.x + particle_index.y * STEP + particle_index.z * STEP * STEP);
		// tuple: (hash, ID)
		uint2 particle_hash_index = make_uint2(particle_hash, tid);
		grid_hash_index[tid] = particle_hash_index;
	}
}

__global__ void find_cell_start_kernel(
	float* position,
	uint2* grid_hash_index,
	int* cell_begin_idx,
	int* cell_end_idx,
	std::size_t num_particles) {

	__shared__ uint2 compare_hash[128 + 1]; //block size + 1 
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < num_particles) {

		compare_hash[threadIdx.x + 1] = grid_hash_index[tid];

		if (threadIdx.x == 0 && tid > 0) {
			compare_hash[0] = grid_hash_index[tid - 1];
		}

		__syncthreads();

		uint2 hash_id_pair;
		uint2 prev_hash_id_pair = compare_hash[threadIdx.x];

		hash_id_pair = compare_hash[threadIdx.x + 1];

		if (tid == 0 || hash_id_pair.x != prev_hash_id_pair.x) {
			cell_begin_idx[hash_id_pair.x] = tid;

			if (tid > 0)
				cell_end_idx[prev_hash_id_pair.x] = tid;
		}
		if (tid == num_particles - 1) {
			cell_end_idx[hash_id_pair.x] = tid + 1;
		}
	}
}

__global__ void update_kernel(
	float* position,
	float3* prev_pos,
	uint2* grid_hash_index,
	int* cell_begin_idx,
	int* cell_end_idx,
	float dt,
	ParticleSystemParameters params,
	std::size_t num_particles) {

	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;


	if (tid < num_particles) {
		float3 pos_t;
		float3 resultant_accln = make_float3(0, 0, 0);
		float3 accln;
		uint2 particle_hash_index;
		

		particle_hash_index = grid_hash_index[tid];
		unsigned int particle_idx = particle_hash_index.y;

		pos_t.x = position[particle_idx * 4 + 0];
		pos_t.y = position[particle_idx * 4 + 1];
		pos_t.z = position[particle_idx * 4 + 2];
		float radius = position[particle_idx * 4 + 3];

		float3 pos_t_1 = prev_pos[particle_idx];
		
		int l = (pos_t.x - params.bb_min[0]) / d_MAX_GRIDSIZE;
		int m = (pos_t.y - params.bb_min[1]) / d_MAX_GRIDSIZE;
		int n = (pos_t.z - params.bb_min[2]) / d_MAX_GRIDSIZE;	

		#pragma unroll
		for (int i = -1; i <= 1; i++) {
			#pragma unroll
			for (int j = -1; j <= 1; j++) {
				#pragma unroll
				for (int k = -1; k <= 1; k++) {
					int3 neighbour_index = make_int3(l + i, m + j, n + k);

					float3 resultant_force = make_float3(0, 0, 0);
					unsigned int neighbour_hash = floorf(neighbour_index.x + neighbour_index.y * STEP + neighbour_index.z * STEP * STEP);
					
					/*
					unsigned int start_idx = cell_begin_idx[neighbour_hash];
					unsigned int end_idx = cell_end_idx[neighbour_hash];
					uint2 particle_b = grid_hash_index[start_idx];

					if (invalid_index(neighbour_index) == true) {
						resultant_force = make_float3(0, 0, 0);
					}
					else {
						for (unsigned int idx = start_idx; idx < end_idx; idx++) {
							particle_b = grid_hash_index[idx];

							float3 pt_b;
							pt_b.x = position[particle_b.y * 4 + 0];
							pt_b.y = position[particle_b.y * 4 + 1];
							pt_b.z = position[particle_b.y * 4 + 2];
							float r_b = position[particle_b.y * 4 + 3];

							float3 pos_t_ba = pt_b - pos_t;
							float dist_ba = sqrtf(pos_t_ba.x * pos_t_ba.x + pos_t_ba.y * pos_t_ba.y + pos_t_ba.z * pos_t_ba.z);

							//equation (1)
							if (dist_ba > 0 && dist_ba < (radius + r_b)) {
								float3 pos_t_1_b = prev_pos[particle_b.y];
								float3 v_a = (pos_t - pos_t_1) / dt; //equation (2)
								float3 v_b = (pt_b - pos_t_1_b) / dt;
								float3 dir_ba = (pos_t_ba) / dist_ba; //equation (3)
								float3 v_ba = v_b - v_a;
								float3 v_ab = v_ba - dir_ba * (v_ba.x * dir_ba.x + v_ba.y * dir_ba.y + v_ba.z * dir_ba.z); //equation (4)
								resultant_force += ((-params.coll_spring * (radius + r_b - dist_ba)) * dir_ba); //equation (5)
								resultant_force += (params.coll_damping * v_ba); //equation (6)
								resultant_force += (params.coll_shear * v_ab); //equation (7)
							}
						}
					}
					
					resultant_accln += resultant_force;
					*/


					resultant_accln += calculate_acceleration(position, prev_pos, grid_hash_index,
						cell_begin_idx, cell_end_idx, neighbour_index,
						pos_t, pos_t_1, radius, params, num_particles, dt);
					
				}
			}
		}

		accln.x = resultant_accln.x + params.gravity[0];
		accln.y = resultant_accln.y + params.gravity[1];
		accln.z = resultant_accln.z + params.gravity[2];

		bool outside;
		outside = check_outside(pos_t_1, radius, params);

		while (outside) {
			reflect_particle(pos_t, pos_t_1, radius, params);
			outside = check_outside(pos_t_1, radius, params);
		}

		float3 pt_pls = (2 * pos_t) - pos_t_1 + (dt * dt * accln);

		position[particle_idx * 4 + 0] = pt_pls.x;
		position[particle_idx * 4 + 1] = pt_pls.y;
		position[particle_idx * 4 + 2] = pt_pls.z;

		prev_pos[particle_idx] = pos_t;
	}

}

void update_particles(
	float* position,
	float3* prev_pos,
	float dt,
	uint2* grid_hash_index, 
	int* cell_begin_idx, 
	int* cell_end_idx,
	ParticleSystemParameters params,
	std::size_t num_particles) {

	constexpr int BLOCK_SIZE = 128;
	auto NUM_BLOCKS = (num_particles + BLOCK_SIZE - 1) / BLOCK_SIZE;

	calculate_hash_kernel <<<NUM_BLOCKS, BLOCK_SIZE, 0>>> (position, grid_hash_index, params, num_particles);
	
	thrust::sort(
		thrust::device_ptr<uint2>(grid_hash_index),
		thrust::device_ptr<uint2>(grid_hash_index + (num_particles)),
		sort_uint2());

	find_cell_start_kernel <<< NUM_BLOCKS, BLOCK_SIZE, 0>>> (
		position, 
		grid_hash_index,
		cell_begin_idx,
		cell_end_idx,
		num_particles);

	update_kernel << <NUM_BLOCKS, BLOCK_SIZE, 0 >> > (
		position,
		prev_pos,
		grid_hash_index,
		cell_begin_idx,
		cell_end_idx,
		dt,
		params,
		num_particles);
}