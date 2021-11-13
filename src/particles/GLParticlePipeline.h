#ifndef INCLUDED_GL_PARTICLE_PIPELINE
#define INCLUDED_GL_PARTICLE_PIPELINE

#pragma once

#include <GL/gl.h>

#include <GL/shader.h>


class GLParticlePipeline
{
	GL::Program particle_prog;

public:
	GLParticlePipeline();

	void draw(GLsizei offset, GLsizei num_particles, GLuint vao) const;
};

#endif  // INCLUDED_GL_PARTICLE_PIPELINE
