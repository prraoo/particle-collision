#ifndef INCLUDED_GLSCENE
#define INCLUDED_GLSCENE

#pragma once

#include <GL/buffer.h>

#include <utils/Camera.h>


class GLScene
{
	const Camera* camera = nullptr;

	GL::Buffer camera_uniform_buffer;

	bool draw_bounding_box = true;

protected:
	virtual void draw(bool bounding_box) const = 0;

	GLScene();

public:
	virtual ~GLScene() = default;

	void attach(const Camera* navigator);

	bool toggle_bounding_box();

	virtual void reset() = 0;
	virtual float update(int steps, float dt) = 0;
	void draw(int viewport_width, int viewport_height);
};

#endif  // INCLUDED_GLSCENE
