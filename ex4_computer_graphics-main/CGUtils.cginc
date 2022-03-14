#ifndef CG_UTILS_INCLUDED
#define CG_UTILS_INCLUDED

#define PI 3.141592653
// A struct containing all the data needed for bump-mapping
struct bumpMapData
{
	float3 normal;       // Mesh surface normal at the point
	float3 tangent;      // Mesh surface tangent at the point
	float2 uv;           // UV coordinates of the point
	sampler2D heightMap; // Heightmap texture to use for bump mapping
	float du;            // Increment size for u partial derivative approximation
	float dv;            // Increment size for v partial derivative approximation
	float bumpScale;     // Bump scaling factor
};


// Receives pos in 3D cartesian coordinates (x, y, z)
// Returns UV coordinates corresponding to pos using spherical texture mapping
float2 getSphericalUV(float3 pos)
{
	// float r = pow(dot(pos.xyz, pos.xyz), 0.5);
	float r = length(pos);
	float theta = atan2(pos.z, pos.x);
	float phi = acos(pos.y / r);
	float u = 0.5 + theta / (2 * PI);
	float v = 1 - phi / PI;
	return float2(u, v);
}

// Implements an adjusted version of the Blinn-Phong lighting model
fixed3 blinnPhong(float3 n, float3 v, float3 l, float shininess, fixed4 albedo, fixed4 specularity, float ambientIntensity)
{
	fixed4 ambient = ambientIntensity * albedo;
	fixed4 diffuse = max(0, dot(n, l)) * albedo;
	float3 h = normalize((l + v) / 2);
	fixed4 specular = pow(max(0, dot(n, h)), shininess) * specularity;
	return ambient + diffuse + specular;
}

// Returns the world-space bump-mapped normal for the given bumpMapData
float3 getBumpMappedNormal(bumpMapData i)
{
	float f_u = (tex2D(i.heightMap, i.uv + float2(i.du, 0)) - tex2D(i.heightMap, i.uv)) / i.du;
	float f_v = (tex2D(i.heightMap, i.uv + float2(0, i.dv)) - tex2D(i.heightMap, i.uv)) / i.dv;
	float3 b = cross(i.tangent, i.normal);
	float3 normal_h = normalize(float3(-i.bumpScale * f_u, -i.bumpScale * f_v, 1));
	float3 n_world = i.tangent * normal_h.x + i.normal * normal_h.z + b * normal_h.y;
	return n_world;
}


#endif // CG_UTILS_INCLUDED
