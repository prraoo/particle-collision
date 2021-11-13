#include <cmath>

#include <GL/error.h>

#include <utils/io/pfm.h>
#include <utils/Camera.h>

#include "GLScene.h"


extern const char env_vs[];
extern const char env_fs[];
extern const char model_vs[];
extern const char model_fs[];

GLScene::GLScene(const Camera& camera, const image2D<std::array<float, 4>>& env, const float* vertex_data, GLsizei num_vertices, const std::uint32_t* index_data, GLsizei num_indices)
	: camera(camera), num_indices(num_indices)
{
	{
		auto vs = GL::compileVertexShader(env_vs);
		auto fs = GL::compileFragmentShader(env_fs);
		glAttachShader(prog_env, vs);
		glAttachShader(prog_env, fs);
		GL::linkProgram(prog_env);
	}

	{
		auto vs = GL::compileVertexShader(model_vs);
		auto fs = GL::compileFragmentShader(model_fs);
		glAttachShader(prog_model, vs);
		glAttachShader(prog_model, fs);
		GL::linkProgram(prog_model);
	}


	glBindBuffer(GL_UNIFORM_BUFFER, camera_uniform_buffer);
	glBufferStorage(GL_UNIFORM_BUFFER, Camera::uniform_buffer_size, nullptr, GL_DYNAMIC_STORAGE_BIT);
	GL::throw_error();


	{
		glBindTexture(GL_TEXTURE_2D, envmap);
		glTexStorage2D(GL_TEXTURE_2D, /*static_cast<GLsizei>(std::log2(std::max(width(env), height(env)))) +*/ 1, GL_RGBA16F, static_cast<GLsizei>(width(env)), static_cast<GLsizei>(height(env)));
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, static_cast<GLsizei>(width(env)), static_cast<GLsizei>(height(env)), GL_RGBA, GL_FLOAT, data(env));
		//glGenerateMipmap(GL_TEXTURE_2D);

		//glSamplerParameteri(envmap_sampler, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT);
		//glSamplerParameteri(envmap_sampler, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT);
		//glSamplerParameteri(envmap_sampler, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
	}
	GL::throw_error();

	if (num_indices)
	{
		glBindVertexArray(vao_model);
		glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);
		glBufferStorage(GL_ARRAY_BUFFER, num_vertices * 6 * 4U, vertex_data, 0U);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_buffer);
		glBufferStorage(GL_ELEMENT_ARRAY_BUFFER, num_indices * 4U, index_data, 0U);

		glBindVertexBuffer(0U, vertex_buffer, 0U, 24U);
		glEnableVertexAttribArray(0U);
		glEnableVertexAttribArray(1U);
		glVertexAttribFormat(0U, 3, GL_FLOAT, GL_FALSE, 0U);
		glVertexAttribFormat(1U, 3, GL_FLOAT, GL_FALSE, 12U);
		glVertexAttribBinding(0U, 0U);
		glVertexAttribBinding(1U, 0U);
		GL::throw_error();
	}
}

void GLScene::draw(int framebuffer_width, int framebuffer_height) const
{
	std::byte camera_uniform_data[Camera::uniform_buffer_size];
	camera.writeUniformBuffer(camera_uniform_data, framebuffer_width * 1.0f / framebuffer_height);
	glBindBufferBase(GL_UNIFORM_BUFFER, 0U, camera_uniform_buffer);
	glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(camera_uniform_data), camera_uniform_data);

	//glEnable(GL_CULL_FACE);
	//glDisable(GL_CULL_FACE);
	//glFrontFace(GL_CCW);

	glClearColor(0.6f, 0.7f, 1.0f, 1.0f);
	glClearDepth(1.0f);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


	glDisable(GL_DEPTH_TEST);
	glDepthMask(GL_FALSE);

	glBindVertexArray(vao_env);
	glUseProgram(prog_env);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, envmap);
	//glBindSampler(0, envmap_sampler);
	glUniform1i(0, 0);

	glDrawArrays(GL_TRIANGLE_STRIP, 0, 3);
	GL::throw_error();


	glEnable(GL_DEPTH_TEST);
	glDepthMask(GL_TRUE);

	glBindVertexArray(vao_model);
	glUseProgram(prog_model);
	glUniform1i(0, 0);
	glDrawElements(GL_TRIANGLES, num_indices, GL_UNSIGNED_INT, nullptr);
	GL::throw_error();
}
