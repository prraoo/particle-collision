#version 430

#include "envmap"

layout(location = 0) uniform sampler2D envmap;

in vec3 d;

layout(location = 0) out vec4 color;

void main()
{
	color = texture(envmap, lat_long(d));
}
