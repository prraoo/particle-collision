#ifndef INCLUDED_GLSCENE
#define INCLUDED_GLSCENE

#pragma once

#include <GL/shader.h>
#include <GL/buffer.h>
#include <GL/texture.h>
#include <GL/vertex_array.h>

#include <utils/Camera.h>
#include <utils/image.h>


class GLScene
{
	GL::VertexArray vao_env;

	GL::VertexArray vao_model;
	GL::Buffer vertex_buffer;
	GL::Buffer index_buffer;

	GL::Program prog_env;
	GL::Program prog_model;

	GL::Buffer camera_uniform_buffer;

	GL::Texture envmap;
	//GL::Sampler envmap_sampler;

	GLsizei num_indices;

	const Camera& camera;

public:
	GLScene(const Camera& camera, const image2D<std::array<float, 4>>& env, const float* vertex_data, GLsizei num_vertices, const std::uint32_t* index_data, GLsizei num_indices);

	void draw(int framebuffer_width, int framebuffer_height) const;
};

#endif  // INCLUDED_GLSCENE
