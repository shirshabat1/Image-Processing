#ifndef CG_RANDOM_INCLUDED
// Upgrade NOTE: excluded shader from DX11 because it uses wrong array syntax (type[size] name)
#pragma exclude_renderers d3d11
#define CG_RANDOM_INCLUDED

// Returns a psuedo-random float between -1 and 1 for a given float c
float random(float c)
{
	return -1.0 + 2.0 * frac(43758.5453123 * sin(c));
}

// Returns a psuedo-random float2 with componenets between -1 and 1 for a given float2 c 
float2 random2(float2 c)
{
	c = float2(dot(c, float2(127.1, 311.7)), dot(c, float2(269.5, 183.3)));

	float2 v = -1.0 + 2.0 * frac(43758.5453123 * sin(c));
	return v;
}

// Returns a psuedo-random float3 with componenets between -1 and 1 for a given float3 c 
float3 random3(float3 c)
{
	float j = 4096.0 * sin(dot(c, float3(17.0, 59.4, 15.0)));
	float3 r;
	r.z = frac(512.0 * j);
	j *= .125;
	r.x = frac(512.0 * j);
	j *= .125;
	r.y = frac(512.0 * j);
	r = -1.0 + 2.0 * r;
	return r.yzx;
}

// Interpolates a given array v of 4 float2 values using bicubic interpolation
// at the given ratio t (a float2 with components between 0 and 1)
//
// [0]=====o==[1]
//         |
//         t
//         |
// [2]=====o==[3]
//
float bicubicInterpolation(float2 v[4], float2 t)
{
	float2 u = t * t * (3.0 - 2.0 * t); // Cubic interpolation

	// Interpolate in the x direction
	float x1 = lerp(v[0], v[1], u.x);
	float x2 = lerp(v[2], v[3], u.x);

	// Interpolate in the y direction and return
	return lerp(x1, x2, u.y);
}

// Interpolates a given array v of 4 float2 values using biquintic interpolation
// at the given ratio t (a float2 with components between 0 and 1)
float biquinticInterpolation(float2 v[4], float2 t)
{
	float2 u = 6 * pow(t, 5) - 15 * pow(t, 4) + 10 * pow(t, 3); // Quintics interpolation

	// Interpolate in the x direction
	float x1 = lerp(v[0], v[1], u.x);
	float x2 = lerp(v[2], v[3], u.x);

	// Interpolate in the y direction and return
	return lerp(x1, x2, u.y);
}

// Interpolates a given array v of 8 float3 values using triquintic interpolation
// at the given ratio t (a float3 with components between 0 and 1)
float triquinticInterpolation(float3 v[8], float3 t)
{
	// Your implementation
	return 0;
}


// topL    =====o==   topR
//              |
//              t
//              |
// bottomL =====o==    bottomR
// Returns the value of a 2D value noise function at the given coordinates c
float value2d(float2 c)
{
	// Calculate the 4 corners of the grid cell containing `c`
	float xMin = floor(c.x);
	float yMin = floor(c.y);
	float xMax = ceil(c.x);
	float yMax = ceil(c.y);
	float2 topL = float2(xMin, yMax);
	float2 bottomL = float2(xMin, yMin);
	float2 topR = float2(xMax, yMax);
	float2 bottomR = float2(xMax, yMin);
	// Use the coordinates the sample the random vectors
	float2 v[4] = {
		 random2(bottomL), random2(bottomR), random2(topL), random2(topR)
	};
	// Take the fractional part of `c` as the interploation parameter
	return bicubicInterpolation(v, frac(c));
}

// Returns the value of a 2D Perlin noise function at the given coordinates c
float perlin2d(float2 c)
{
	// Calculate the 4 corners of the grid cell containing `c`
	float xMin = floor(c.x);
	float yMin = floor(c.y);
	float xMax = ceil(c.x);
	float yMax = ceil(c.y);

	float2 topL = float2(xMin, yMax);
	float2 bottomL = float2(xMin, yMin);
	float2 topR = float2(xMax, yMax);
	float2 bottomR = float2(xMax, yMin);

	// Sample 4 random vectors that represent gradient vectors
	float2 randBottomL = random2(bottomL);
	float2 randTopL = random2(topL);
	float2 randTopR = random2(topR);
	float2 randBottomR = random2(bottomR);

	// Calculate 4 distance vectors from the point `c` to the grid cell corners
	float2 distBottomL = c - bottomL;
	float2 distTopL = c - topL;
	float2 distTopR = c - topR;
	float2 distBottomR = c - bottomR;

	// Calculate the dot product of the respective random gradients and distance vectors to get 4 influence values
	float2 dotBottomL = dot(randBottomL, distBottomL);
	float2 dotTopL = dot(randTopL, distTopL);
	float2 dotTopR = dot(randTopR, distTopR);
	float2 dotBottomR = dot(randBottomR, distBottomR);

	float2 v[4] = {
		dotBottomL, dotBottomR, dotTopL, dotTopR
	};
	return biquinticInterpolation(v, frac(c));

}

// Returns the value of a 3D Perlin noise function at the given coordinates c
float perlin3d(float3 c)
{
	// Your implementation
	return 0;
}


#endif // CG_RANDOM_INCLUDED
