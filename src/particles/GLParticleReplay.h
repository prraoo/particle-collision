#ifndef INCLUDED_GL_PARTICLE_REPLAY
#define INCLUDED_GL_PARTICLE_REPLAY

#pragma once

#include <cstdint>
#include <memory>
#include <vector>
#include <chrono>

#include <GL/gl.h>

#include <GL/vertex_array.h>
#include <GL/buffer.h>

#include "particle_system_state.h"

#include "GLScene.h"

#include "GLParticlePipeline.h"
#include "GLBoundingBox.h"


class GLParticleReplayBuilder : public virtual ParticleReplayBuilder
{
	std::size_t num_particles;
	std::vector<std::chrono::nanoseconds> t;
	std::vector<float> position_data;
	std::vector<std::uint32_t> color_data;
	math::float3 bb_min;
	math::float3 bb_max;

	void add_frame(std::chrono::nanoseconds dt, const float* position, const std::uint32_t* color) override;

public:
	GLParticleReplayBuilder(std::size_t num_particles, const float* position, const std::uint32_t* color, const ParticleSystemParameters& params);

	std::tuple<std::unique_ptr<GLScene>, math::float3, math::float3> finish();
};


class GLParticleReplay : public virtual GLScene
{
	GLsizei num_particles;
	std::chrono::nanoseconds time = std::chrono::nanoseconds(0);
	int frame = 0;

	GLParticlePipeline pipeline;

	GL::VertexArray particles_vao;

	GLBoundingBox bounding_box;

	std::vector<std::chrono::nanoseconds> frame_times;

	GL::Buffer particle_position_buffer;
	GL::Buffer particle_color_buffer;

public:
	GLParticleReplay(std::size_t num_particles, std::vector<std::chrono::nanoseconds> frame_times, const float* position, const std::uint32_t* color, const math::float3& bb_min, const math::float3& bb_max);

	void reset() override;
	float update(int steps, float dt) override;
	void draw(bool draw_bounding_box) const override;
};

#endif  // INCLUDED_GL_PARTICLE_REPLAY
