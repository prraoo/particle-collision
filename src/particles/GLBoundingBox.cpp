#include <GL/error.h>

#include "GLBoundingBox.h"


extern const char bounding_box_vs[];
extern const char bounding_box_fs[];

GLBoundingBox::GLBoundingBox(const math::float3& bb_min, const math::float3& bb_max)
{
	{
		auto vs = GL::compileVertexShader(bounding_box_vs);
		auto fs = GL::compileFragmentShader(bounding_box_fs);
		glAttachShader(prog, vs);
		glAttachShader(prog, fs);
		GL::linkProgram(prog);
	}

	glProgramUniform3f(prog, 0, bb_max.x - bb_min.x, bb_max.y - bb_min.y, bb_max.z - bb_min.z);
	glProgramUniform3f(prog, 1, (bb_min.x + bb_max.x) * 0.5f, (bb_min.y + bb_max.y) * 0.5f, (bb_min.z + bb_max.z) * 0.5f);

	{
		//GLfloat vertices[] = {
		//	-1.0f, -1.0f, -1.0f,
		//	 1.0f, -1.0f, -1.0f,
		//	-1.0f,  1.0f, -1.0f,
		//	 1.0f,  1.0f, -1.0f,
		//	-1.0f, -1.0f,  1.0f,
		//	 1.0f, -1.0f,  1.0f,
		//	-1.0f,  1.0f,  1.0f,
		//	 1.0f,  1.0f,  1.0f,
		//};

		GLushort indices[] = {
			6, 0, 4,
			0, 6, 2,

			5, 3, 7,
			3, 5, 1,

			1, 2, 3,
			2, 1, 0,

			4, 7, 6,
			7, 4, 5,

			0, 5, 4,
			5, 0, 1,

			6, 3, 2,
			3, 6, 7
		};

		glBindVertexArray(vao);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_buffer);
		glBufferStorage(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, 0U);
	}

	GL::throw_error();
}

void GLBoundingBox::draw() const
{
	glEnable(GL_BLEND);
	glDepthMask(GL_FALSE);
	glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
	//glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

	glBindVertexArray(vao);
	glUseProgram(prog);
	glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_SHORT, nullptr);

	//glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

	GL::throw_error();
}
