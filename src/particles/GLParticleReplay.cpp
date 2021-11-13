#include <algorithm>

#include <GL/error.h>

#include "GLParticleReplay.h"


namespace
{
	std::vector<float> init_position(std::size_t num_particles, const float* position)
	{
		std::vector<float> position_data;

		position_data.reserve(4 * num_particles);

		for (std::size_t i = 0; i < num_particles; ++i)
		{
			position_data.push_back(position[0 * num_particles + i]);
			position_data.push_back(position[1 * num_particles + i]);
			position_data.push_back(position[2 * num_particles + i]);
			position_data.push_back(position[3 * num_particles + i]);
		}

		return position_data;
	}
}

void GLParticleReplayBuilder::add_frame(std::chrono::nanoseconds dt, const float* position, const std::uint32_t* color)
{
	t.push_back(t.back() + dt);
	position_data.insert(std::end(position_data), position, position + 4 * num_particles);
	color_data.insert(std::end(color_data), color, color + num_particles);
}

GLParticleReplayBuilder::GLParticleReplayBuilder(std::size_t num_particles, const float* position, const std::uint32_t* color, const ParticleSystemParameters& params)
	: num_particles(num_particles),
	  t { std::chrono::nanoseconds(0) },
	  position_data(init_position(num_particles, position)),
	  color_data { color, color + num_particles },
	  bb_min(params.bb_min[0], params.bb_min[1], params.bb_min[2]),
	  bb_max(params.bb_max[0], params.bb_max[1], params.bb_max[2])
{
}

std::tuple<std::unique_ptr<GLScene>, math::float3, math::float3> GLParticleReplayBuilder::finish()
{
	return {
		std::make_unique<GLParticleReplay>(num_particles, std::move(t), &position_data[0], &color_data[0], bb_min, bb_max),
		bb_min,
		bb_max
	};
}


GLParticleReplay::GLParticleReplay(std::size_t num_particles, std::vector<std::chrono::nanoseconds> frame_times, const float* position, const std::uint32_t* color, const math::float3& bb_min, const math::float3& bb_max)
	: num_particles(static_cast<GLsizei>(num_particles)),
	  bounding_box(bb_min, bb_max),
	  frame_times(std::move(frame_times))
{
	glBindBuffer(GL_ARRAY_BUFFER, particle_position_buffer);
	glBufferStorage(GL_ARRAY_BUFFER, num_particles * size(this->frame_times) * 16U, position, 0U);

	glBindBuffer(GL_ARRAY_BUFFER, particle_color_buffer);
	glBufferStorage(GL_ARRAY_BUFFER, num_particles * size(this->frame_times) * 4U, color, 0U);

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

void GLParticleReplay::reset()
{
	time = std::chrono::nanoseconds(0);
	frame = 0;
}

float GLParticleReplay::update(int steps, float dt)
{
	time += steps * std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration<float>(dt));
	auto found = std::upper_bound(begin(frame_times) + frame, end(frame_times), time);

	if (found != end(frame_times))
		frame = found - begin(frame_times) - 1;

	return 0.0f;
}

void GLParticleReplay::draw(bool draw_bounding_box) const
{
	pipeline.draw(frame * num_particles, num_particles, particles_vao);

	if (draw_bounding_box)
		bounding_box.draw();
}
