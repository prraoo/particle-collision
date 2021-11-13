#version 430

layout(location = 0) out vec4 color;


in vec2 pos;
in vec4 albedo;

void main()
{
	float d2 = 1.0f - dot(pos, pos);

	if (d2 < 0.0f)
		discard;

	float d = sqrt(d2);

	color = albedo * vec4(d, d, d, 1.0f);
}
