#version 430

#include <camera>

layout(location = 0) in vec3 v_p;
layout(location = 1) in vec3 v_n;

out vec3 a_p;
out vec3 a_n;

void main()
{
	gl_Position = camera.PV * vec4(v_p, 1.0f);
	a_p = v_p;
	//a_n = (vec4(v_n, 0.0f) * camera.V_inv).xyz;
	a_n = v_n;
}
