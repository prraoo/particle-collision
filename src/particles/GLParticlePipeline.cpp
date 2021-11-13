#include <GL/error.h>

#include "GLParticlePipeline.h"


extern const char particle_vs[];
extern const char particle_gs[];
extern const char particle_fs[];

GLParticlePipeline::GLParticlePipeline()
{
	{
		auto vs = GL::compileVertexShader(particle_vs);
		auto gs = GL::compileGeometryShader(particle_gs);
		auto fs = GL::compileFragmentShader(particle_fs);
		glAttachShader(particle_prog, vs);
		glAttachShader(particle_prog, gs);
		glAttachShader(particle_prog, fs);
		GL::linkProgram(particle_prog);
	}

	GL::throw_error();
}

void GLParticlePipeline::draw(GLsizei offset, GLsizei num_particles, GLuint vao) const
{
	glDisable(GL_BLEND);

	glBindVertexArray(vao);
	glUseProgram(particle_prog);
	glDrawArrays(GL_POINTS, offset, num_particles);

	GL::throw_error();
}
