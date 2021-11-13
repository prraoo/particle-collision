#include <utils/dynamic_library.h>

#include <utils/CUDA/error.h>

#include "particle_system_state.h"

#include "CUDAParticles.h"


CUDAParticles::CUDAParticles(particle_system_module& module, std::size_t num_particles, std::unique_ptr<float[]> position, std::unique_ptr<std::uint32_t[]> color, const ParticleSystemParameters& params)
	: particles(module.create_instance(num_particles, &position[0] + 0 * num_particles, &position[0] + 1 * num_particles, &position[0] + 2 * num_particles, &position[0] + 3 * num_particles, &color[0], params)),
	  num_particles(num_particles),
	  position_buffer(CUDA::malloc<float>(4 * num_particles)),
	  color_buffer(CUDA::malloc<std::uint32_t>(num_particles)),
	  particles_begin(CUDA::create_event()),
	  particles_end(CUDA::create_event()),
	  position_download_buffer(new float[ 4 * num_particles]),
	  color_download_buffer(new std::uint32_t[num_particles])
{
}

float CUDAParticles::update(std::ostream& file, ParticleReplayWriter& writer, int steps, float dt)
{
	throw_error(cudaEventRecord(particles_begin.get()));

	for (int i = 0; i < steps; ++i)
		particles.update(position_buffer.get(), color_buffer.get(), dt);

	throw_error(cudaEventRecord(particles_end.get()));

	throw_error(cudaEventSynchronize(particles_end.get()));

	throw_error(cudaMemcpy(position_download_buffer.get(), position_buffer.get(), 4 * num_particles * sizeof(float), cudaMemcpyDeviceToHost));
	throw_error(cudaMemcpy(color_download_buffer.get(), color_buffer.get(), num_particles * sizeof(std::uint32_t), cudaMemcpyDeviceToHost));

	writer.add_frame(file, steps * std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration<float>(dt)), position_download_buffer.get(), color_download_buffer.get());

	return CUDA::elapsed_time(particles_begin.get(), particles_end.get());
}
