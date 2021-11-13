#version 430

#include <camera>

layout(location = 0) out vec4 color;

in vec3 l;

void main()
{
	vec3 n = vec3(
		gl_PrimitiveID < 4 ? 1.0f : 0.0f,
		gl_PrimitiveID >= 8 ? 1.0f : 0.0f,
		gl_PrimitiveID >= 4 && gl_PrimitiveID < 8 ? 1.0f : 0.0f
	);

	if (gl_PrimitiveID % 4 > 1)
		n = -n;

	float f = mix(1.0f, 0.3f, max(dot(normalize(l), n), 0.0f));

	color = vec4(f * vec3(0.8f, 0.8f, 0.8f), f);
}
