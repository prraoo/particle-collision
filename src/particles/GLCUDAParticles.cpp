#include <utility>

#include <GL/error.h>

#include <cuda_gl_interop.h>

#include <utils/CUDA/error.h>
#include <utils/CUDA/device.h>
#include <utils/CUDA/graphics_gl_interop.h>

#include "GLCUDAParticles.h"


GLCUDAParticles::GLCUDAParticles(particle_system_instance particles, std::size_t num_particles, std::unique_ptr<float[]> position, std::unique_ptr<std::uint32_t[]> color, const math::float3& bb_min, const math::float3& bb_max)
	: particles(std::move(particles)),
	  bounding_box(bb_min, bb_max),
	  num_particles(static_cast<GLsizei>(num_particles)),
	  particles_begin(CUDA::create_event()),
	  particles_end(CUDA::create_event()),
	  initial_position(std::move(position)),
	  initial_color(std::move(color))
{
	glBindBuffer(GL_ARRAY_BUFFER, particle_position_buffer);
	glBufferStorage(GL_ARRAY_BUFFER, num_particles * 16U, nullptr, 0U);
	particle_position_buffer_resource = CUDA::graphics::register_GL_buffer(particle_position_buffer, cudaGraphicsMapFlagsReadOnly);

	glBindBuffer(GL_ARRAY_BUFFER, particle_color_buffer);
	glBufferStorage(GL_ARRAY_BUFFER, num_particles * 4U, nullptr, 0U);
	particle_color_buffer_resource = CUDA::graphics::register_GL_buffer(particle_color_buffer, cudaGraphicsMapFlagsReadOnly);

	GL::throw_error();

	glBindVertexArray(particles_vao);
	glBindVertexBuffer(0U, particle_position_buffer, 0U, 16U);
	glEnableVertexAttribArray(0U);
	glVertexAttribBinding(0U, 0U);
	glVertexAttribFormat(0U, 4, GL_FLOAT, GL_FALSE, 0U);
	glBindVertexBuffer(1U, particle_color_buffer, 0U, 4U);
	glEnableVertexAttribArray(1U);
	glVertexAttribBinding(1U, 1U);
	glVertexAttribFormat(1U, 4, GL_UNSIGNED_BYTE, GL_TRUE, 0U);

	GL::throw_error();
}

void GLCUDAParticles::reset()
{
	particles.reset(&initial_position[0] + 0 * num_particles, &initial_position[0] + 1 * num_particles, &initial_position[0] + 2 * num_particles, &initial_position[0] + 3 * num_particles, &initial_color[0]);
}

float GLCUDAParticles::update(int steps, float dt)
{
	auto mapped_resources = CUDA::graphics::map_resources(particle_position_buffer_resource.get(), particle_color_buffer_resource.get());

	auto mapped_positions = CUDA::graphics::get_mapped_buffer(particle_position_buffer_resource.get());
	auto mapped_colors = CUDA::graphics::get_mapped_buffer(particle_color_buffer_resource.get());

	throw_error(cudaEventRecord(particles_begin.get()));

	for (int i = 0; i < steps; ++i)
		particles.update(static_cast<float*>(mapped_positions.ptr), static_cast<std::uint32_t*>(mapped_colors.ptr), dt);

	throw_error(cudaEventRecord(particles_end.get()));

	throw_error(cudaEventSynchronize(particles_end.get()));

	return CUDA::elapsed_time(particles_begin.get(), particles_end.get());
}

void GLCUDAParticles::draw(bool draw_bounding_box) const
{
	pipeline.draw(0, num_particles, particles_vao);

	if(draw_bounding_box)
		bounding_box.draw();
}
