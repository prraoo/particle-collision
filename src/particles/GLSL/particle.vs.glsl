#version 430

layout(location = 0) in vec4 position;
layout(location = 1) in vec4 color;


out VS_OUTPUT {
	vec4 position;
	vec4 color;
} vs_out;

void main()
{
	vs_out.position = position;
	vs_out.color = color;
}
