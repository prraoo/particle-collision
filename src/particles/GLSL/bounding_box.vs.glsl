#version 430

#include <camera>

layout(location = 0) uniform vec3 size;
layout(location = 1) uniform vec3 offset;

out vec3 l;

void main()
{
	vec3 p = vec3(
		(gl_VertexID & 1) != 0 ? 0.5f : -0.5f,
		(gl_VertexID & 2) != 0 ? 0.5f : -0.5f,
		(gl_VertexID & 4) != 0 ? 0.5f : -0.5f
	) * size + offset;

	gl_Position = camera.PV * vec4(p, 1.0f);
	l = camera.position - p;
}
