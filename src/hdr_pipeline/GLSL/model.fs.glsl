#version 430

#include <camera>

#include "envmap"

layout(location = 0) uniform sampler2D envmap;
layout(location = 1) uniform vec3 albedo = vec3(0.1f, 0.1f, 0.1f);

in vec3 a_p;
in vec3 a_n;

layout(location = 0) out vec4 color;

void main()
{
	vec3 n = normalize(a_n);
	vec3 v = normalize(camera.position - a_p);
	vec3 r = reflect(-v, n);

	float lambert = max(dot(n, v), 0.0f);

	float bla = 1.0f - lambert;
	float bla2 = bla * bla;

	const float R_0 = 0.14f;

	float R = R_0 + (1 - R_0) * bla2 * bla2 * bla;

	color = vec4(R * texture(envmap, lat_long(r)).rgb + (1.0f - R) * albedo * lambert, 1.0f);
}
