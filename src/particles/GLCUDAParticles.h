#ifndef INCLUDED_GL_CUDA_PARTICLES
#define INCLUDED_GL_CUDA_PARTICLES

#pragma once

#include <cstddef>
#include <cstdint>

#include <GL/gl.h>

#include <GL/shader.h>
#include <GL/vertex_array.h>
#include <GL/buffer.h>

#include <cuda_runtime_api.h>

#include <utils/CUDA/event.h>
#include <utils/CUDA/graphics_interop.h>

#include <utils/dynamic_library.h>

#include "GLScene.h"

#include "particle_system_module.h"

#include "ParticleSystemLoader.h"
#include "GLParticlePipeline.h"
#include "GLBoundingBox.h"


class GLCUDAParticles : public virtual GLScene
{
	particle_system_instance particles;

	GLsizei num_particles;

	GLParticlePipeline pipeline;

	GL::VertexArray particles_vao;

	GLBoundingBox bounding_box;

	GL::Buffer particle_position_buffer;
	GL::Buffer particle_color_buffer;

	CUDA::graphics::unique_resource particle_position_buffer_resource;
	CUDA::graphics::unique_resource particle_color_buffer_resource;

	CUDA::unique_event particles_begin;
	CUDA::unique_event particles_end;

	std::unique_ptr<float[]> initial_position;
	std::unique_ptr<std::uint32_t[]> initial_color;

public:
	GLCUDAParticles(particle_system_instance particles, std::size_t num_particles, std::unique_ptr<float[]> position, std::unique_ptr<std::uint32_t[]> color, const math::float3& bb_min, const math::float3& bb_max);

	void reset() override;
	float update(int steps, float dt) override;
	void draw(bool draw_bounding_box) const override;
};

#endif  // INCLUDED_GL_CUDA_PARTICLES
