#ifndef INCLUDED_GL_BOUNDING_BOX
#define INCLUDED_GL_BOUNDING_BOX

#pragma once

#include <GL/gl.h>

#include <GL/shader.h>
#include <GL/vertex_array.h>
#include <GL/buffer.h>

#include <utils/math/vector.h>


class GLBoundingBox
{
	GL::VertexArray vao;

	GL::Program prog;

	GL::Buffer index_buffer;

public:
	GLBoundingBox(const math::float3& bb_min, const math::float3& bb_max);

	void draw() const;
};

#endif  // INCLUDED_GL_BOUNDING_BOX
