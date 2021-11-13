#include <GL/error.h>

#include "GLScene.h"


GLScene::GLScene()
{
	glBindBuffer(GL_UNIFORM_BUFFER, camera_uniform_buffer);
	glBufferStorage(GL_UNIFORM_BUFFER, Camera::uniform_buffer_size, nullptr, GL_DYNAMIC_STORAGE_BIT);

	GL::throw_error();
}

void GLScene::draw(int viewport_width, int viewport_height)
{
	if (!camera)
		return;

	std::byte camera_uniform_data[Camera::uniform_buffer_size];
	camera->writeUniformBuffer(camera_uniform_data, viewport_width * 1.0f / viewport_height);
	glBindBufferBase(GL_UNIFORM_BUFFER, 0U, camera_uniform_buffer);
	glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(camera_uniform_data), camera_uniform_data);

	draw(draw_bounding_box);
}

bool GLScene::toggle_bounding_box()
{
	return draw_bounding_box = !draw_bounding_box;
}

void GLScene::attach(const Camera* camera)
{
	this->camera = camera;
}
