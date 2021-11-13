#version 430

#include <camera>

out vec3 d;

void main()
{
	vec2 p = vec2((gl_VertexID & 0x2) * 0.5f, (gl_VertexID & 0x1));
	gl_Position = vec4(p * 4.0f - 1.0f, 1.0f, 1.0f);

	vec4 p_2 = camera.PV_inv * vec4(gl_Position.xy, 0.0f, 1.0f);
	vec4 p_1 = camera.PV_inv * vec4(gl_Position.xy, -1.0f, 1.0f);

	d = p_2.xyz / p_2.w - p_1.xyz / p_1.w;
}
