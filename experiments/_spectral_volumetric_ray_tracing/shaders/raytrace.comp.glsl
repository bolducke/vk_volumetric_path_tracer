// Copyright 2020 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0
#version 460 
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_ray_query : require

layout(local_size_x = 16, local_size_y = 8, local_size_z = 1) in;

#define MGRAYDIFFUSE 0
#define MLIGHT 1
#define MGLOSSYREFLECTION 2
#define MWEIRD 3
#define MREDDIFFUSE 4
#define MGREENDIFFUSE 5
#define MBLUEDIFFUSE 6

// The scalar layout qualifier here means to align types according to the alignment
// of their scalar components, instead of e.g. padding them to std140 rules.
layout(binding = 0, set = 0, scalar) buffer storageBuffer
{
  vec3 imageData[];
};
layout(binding = 1, set = 0) uniform accelerationStructureEXT tlas;

layout(binding = 2, set = 0, scalar) buffer Vertices
{
  vec3 vertices[];
};
layout(binding = 3, set = 0, scalar) buffer Indices
{
  uint indices[];
};
layout(binding = 4, set = 0, scalar) buffer Materials
{
  uint materials[];
};

struct Ray {
  vec3 o;
  vec3 d;
};

struct Intersection
{
  vec3 wP;
  vec3 wN;
  uint material;
  float t;
};

// // Returns the color of the sky in a given direction (in linear color space)
// vec3 background(vec3 direction)
// {
//   // +y in world space is up, so:
//   if(direction.y > 0.0f)
//   {
//     return mix(vec3(1.0f), vec3(0.25f, 0.5f, 1.0f), direction.y);
//   }
//   else
//   {
//     return vec3(0.03f);
//   }
// }

vec3 background(vec3 direction) {
  return vec3(0.01);
}

Intersection hit(rayQueryEXT rayQuery)
{
  Intersection result;
  // Get the ID of the triangle
  const int primitiveID = rayQueryGetIntersectionPrimitiveIndexEXT(rayQuery, true);

  // Get the indices of the vertices of the triangle
  const uint i0 = indices[3 * primitiveID + 0];
  const uint i1 = indices[3 * primitiveID + 1];
  const uint i2 = indices[3 * primitiveID + 2];

  // Get the vertices of the triangle
  const vec3 v0 = vertices[i0];
  const vec3 v1 = vertices[i1];
  const vec3 v2 = vertices[i2];

  // Get the barycentric coordinates of the intersection
  vec3 barycentrics = vec3(0.0, rayQueryGetIntersectionBarycentricsEXT(rayQuery, true));
  barycentrics.x    = 1.0 - barycentrics.y - barycentrics.z;

  // Compute the coordinates of the intersection
  const vec3 objectPos = v0 * barycentrics.x + v1 * barycentrics.y + v2 * barycentrics.z;
  // For the main tutorial, object space is the same as world space:
  result.wP = objectPos;

  // Compute the normal of the triangle in object space, using the right-hand rule:
  //    v2      .
  //    |\      .
  //    | \     .
  //    |/ \    .
  //    /   \   .
  //   /|    \  .
  //  L v0---v1 .
  // n
  const vec3 objectNormal = normalize(cross(v1 - v0, v2 - v0));
  // For the main tutorial, object space is the same as world space:
  result.wN = objectNormal;

  result.material = materials[primitiveID];

  result.t = rayQueryGetIntersectionTEXT(rayQuery, true);

  return result;
}

// SAMPLING

// Random number generation using pcg32i_random_t, using inc = 1. Our random state is a uint.
uint stepRNG(uint rng_state)
{
  return rng_state * 747796405 + 1;
}

// Steps the RNG and returns a floating-point value between 0 and 1 inclusive.
float sample_next(inout uint rng_state)
{
  // Condensed version of pcg_output_rxs_m_xs_32_32, with simple conversion to floating-point [0,1].
  rng_state  = stepRNG(rng_state);
  uint word = ((rng_state >> ((rng_state >> 28) + 4)) ^ rng_state) * 277803737;
  word      = (word >> 22) ^ word;
  return float(word) / 4294967295.0f;
}

vec3 sample_hemisphere(inout uint rng_state) {
  vec3 p;
  const float theta = 6.2831853 * sample_next(rng_state);   // Random in [0, 2pi]
  const float u     = 2.0 * sample_next(rng_state) - 1.0;  // Random in [-1, 1]
  const float r     = sqrt(1.0 - u * u);
  p = vec3(r * cos(theta), r * sin(theta), u);
  return p;
}

vec3 sample_spherical(inout uint rng_state) {
  float r = 1;
  float theta = acos(2.0 * sample_next(rng_state) - 1); // asin(2 )
  float phi = 2 * 3.14 * sample_next(rng_state);


  float x = r * sin(theta) * cos(phi);
  float y = r * sin(theta) * sin(phi);
  float z = r * cos(theta);

  return vec3(x,y,z);
}

// MATERIALS //

// Sample a new diffuse direction
vec3 sample_diffuse(vec3 r_d, Intersection its, inout uint rng_state) {
    // For a random diffuse bounce direction, we follow the approach of
    // Ray Tracing in One Weekend, and generate a random point on a sphere
    // of radius 1 centered at the normal. This uses the random_unit_vector
    // function from chapter 8.5:
    vec3 p = sample_hemisphere(rng_state);
    r_d = its.wN + p;
    r_d = normalize(r_d); 
    return r_d;
}

// Sample a new diffuse direction
vec3 sample_glossy(vec3 r_d, Intersection its, inout uint rng_state) {
    // For a random diffuse bounce direction, we follow the approach of
    // Ray Tracing in One Weekend, and generate a random point on a sphere
    // of radius 1 centered at the normal. This uses the random_unit_vector
    // function from chapter 8.5:
    vec3 p = sample_hemisphere(rng_state);
    r_d = reflect(r_d, its.wN) + 0.02 * p;
    r_d = normalize(r_d); 
    return r_d;
}

vec3 Tr(vec3 sigmat, float t) {
  return exp(-sigmat * t);
}

vec3 sample_Tr(vec3 sigmat, inout uint rng_state) {
  return -log(1.0 - sample_next(rng_state)) / sigmat;
}

// PHASE FUNCTION //

vec3 frame_to_world(vec3 n, vec3 v) {

    float s = sign(n.z);

    float a = -1.0 / (s + n.z);
    float b = n.x * n.y * a;
    vec3 x = vec3(1.0 + s * n.x * n.x * a, s * b, -s * n.x);
    vec3 y = vec3(b, s + n.y * n.y * a, -n.y);
    vec3 z = n;

    return x* v.x + y * v.y + z * v.z;
}

vec3 sample_fp_iso(inout uint rng_state) {
  return sample_spherical(rng_state);
}

vec3 sample_fp_hg(float g, vec3 d_in, inout uint rng_state) {

  // We sample in every direction possible
  if ( abs(g) < 0.001) {
    return sample_spherical(rng_state);
  }

	float cos_theta = 1.0/(2.0 * g) * 
						(1 + pow(g,2) - pow((1-pow(g,2))/
											(1 - g + 2 * g * sample_next(rng_state)),2));

	float theta = acos(cos_theta);
	float phi = 2 * 3.14 * sample_next(rng_state);

	return frame_to_world(d_in, vec3(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta)));
}

void main()
{
  const int maxdepth = 128;

  //media parameter
  const float g = 0.5;
  const float density = 0.01;
  const vec3 color = vec3(1.0, 1.0, 0.3);

  vec3 sigmat; vec3 sigmaa; vec3 sigmas;
  sigmaa = vec3(0.0); // ALWAYS 0 in our context
  sigmas = color * density;
  sigmat = color * density;

  // The resolution of the buffer, which in this case is a hardcoded vector
  // of 2 unsigned integers:
  const uvec2 resolution = uvec2(800, 600);

  // Get the coordinates of the pixel for this invocation:
  //
  // .-------.-> x
  // |       |
  // |       |
  // '-------'
  // v
  // y
  const uvec2 pixel = gl_GlobalInvocationID.xy;

  // If the pixel is outside of the image, don't do anything:
  if((pixel.x >= resolution.x) || (pixel.y >= resolution.y))
  {
    return;
  }

  // State of the random number generator.
  uint rng_state = resolution.x * pixel.y + pixel.x;  // Initial seed

  // This scene uses a right-handed coordinate system like the OBJ file format, where the
  // +x axis points right, the +y axis points up, and the -z axis points into the screen.
  // The camera is located at (-0.001, 1, 6).
  const vec3 cameraOrigin = vec3(-0.001, 1.0, 6.0);
  // Define the field of view by the vertical slope of the topmost rays:
  const float fovVerticalSlope = 1.0 / 5.0;

  // The sum of the colors of all of the samples.
  vec3 Lo = vec3(0.0);

  Ray r;

  // Limit the kernel to trace at most 64 samples.
  const int BATCH_SAMPLE = 12;
  const int NUM_SAMPLES = 256;
  for(int batch_sample = 0; batch_sample < BATCH_SAMPLE; batch_sample++) {

    for(int sampleIdx = 0; sampleIdx < NUM_SAMPLES; sampleIdx++)
    {
      // Rays always originate at the camera for now. In the future, they'll
      // bounce around the scene.
      r.o = cameraOrigin;
      // Compute the direction of the ray for this pixel. To do this, we first
      // transform the screen coordinates to look like this, where a is the
      // aspect ratio (width/height) of the screen:
      //           1
      //    .------+------.
      //    |      |      |
      // -a + ---- 0 ---- + a
          // throughput = vec3(0.0, 0.0, 1.0);
          // break;
      //    |      |      |
      //    '------+------'
      //          -1
      const vec2 randomPixelCenter = vec2(pixel) + vec2(sample_next(rng_state), sample_next(rng_state));
      const vec2 screenUV          = vec2((2.0 * randomPixelCenter.x - resolution.x) / resolution.y,    //
                                -(2.0 * randomPixelCenter.y - resolution.y) / resolution.y);  // Flip the y axis
      // Create a ray direction:
      r.d = vec3(fovVerticalSlope * screenUV.x, fovVerticalSlope * screenUV.y, -1.0);
      r.d      = normalize(r.d);

      vec3 throughput = vec3(1.0);  // The amount of light that made it to the end of the current ray.

      // Limit the kernel to trace at most 32 segments.
      for(int depth = 0; depth < maxdepth; depth++)
      {
        // Trace the ray and see if and where it intersects the scene!
        // First, initialize a ray query object:
        rayQueryEXT rayQuery;
        rayQueryInitializeEXT(rayQuery,              // Ray query
                              tlas,                  // Top-level acceleration structure
                              gl_RayFlagsOpaqueEXT,  // Ray flags, here saying "treat all geometry as opaque"
                              0xFF,                  // 8-bit instance mask, here saying "trace against all instances"
                              r.o,             // Ray origin
                              0.0,                   // Minimum t-value
                              r.d,          // Ray direction
                              10000.0);              // Maximum t-value

        // Start traversal, and loop over all ray-scene intersections. When this finishes,
        // rayQuery stores a "committed" intersection, the closest intersection (if any).
        while(rayQueryProceedEXT(rayQuery))
        {
        }

        // Get the type of committed (true) intersection - nothing, a triangle, or
        // a generated object
        if(rayQueryGetIntersectionTypeEXT(rayQuery, true) == gl_RayQueryCommittedIntersectionTriangleEXT)
        {
          Intersection its = hit(rayQuery);

          uint idx_c = min(int(sample_next(rng_state) * 3), 2);

          float t = sample_Tr(sigmat, rng_state)[idx_c];

          if (t < its.t) {
            vec3 x = r.o + t * r.d;	// Point in the media
            
            float Pac = sigmaa[idx_c] / sigmat[idx_c]; // Equivalent to 0
            float Psc = sigmas[idx_c] / sigmat[idx_c];

            float event_choice = sample_next(rng_state);

            // * ft(t) / p(t) = 1 when every spectre are equals
            vec3 tmp = (sigmat * Tr(sigmat, t));
            throughput *= 3.0*tmp / (tmp.x + tmp.y + tmp.z);

            if (event_choice < Pac){
              // Participing Media is Emitting No Light. Still, if we hit an absorption particles, we terminate the path.
              throughput = vec3(0.0);
              break;
            } else { // event_choice < [(Pac + Psc) == 1.0]
              vec3 nw = normalize(sample_fp_hg(g, r.d, rng_state));
              r = Ray(x, nw);

              // * (m_sigma_s/m_sigma_t)/Psc * fp(x, w, w') / p(w'|x, w)
              // fp(x, w, w') / p(w'|x, w) = 1 (We sample the direction from the same distribution as fp)
              // (m_sigma_s/m_sigma_t)/Psc = 1 when every spectre are equals
              throughput *= (sigmas/sigmat)/Psc;
            }
          }
          else {

              //hmm?
            its.wN = faceforward(its.wN, r.d, its.wN);

            //Either way, because we hit the surface, we need to attenuate the throughput based on the transmittance
            // Tr(z) / P(z) = 1 when every spectre are equals
            vec3 tmp = (Tr(sigmat, its.t));
            throughput *= 3.0*tmp / (tmp.x + tmp.y + tmp.z);

            if(its.material == MGRAYDIFFUSE) {
              // Start a new ray at the hit position, but offset it slightly along
              // the normal against r_d:
              r.o = its.wP - 0.0001 * sign(dot(r.d, its.wN)) * its.wN;
              r.d = sample_diffuse(r.d, its, rng_state);

              throughput *= vec3(0.7);
            } else if (its.material == MLIGHT) {
              // 1 is a Light Material
              if ( dot(r.d, its.wN) > 0) {
                throughput *= vec3(0.0);
              } else {
                throughput *= vec3(10.0);
              }

              Lo += throughput;
              break;
            } else if (its.material == MREDDIFFUSE) {
              r.o = its.wP - 0.0001 * sign(dot(r.d, its.wN)) * its.wN;
              r.d = sample_diffuse(r.d, its, rng_state);

              throughput *= vec3(0.7, 0.0, 0.0);
            } else if(its.material == MGREENDIFFUSE) {
              r.o = its.wP - 0.0001 * sign(dot(r.d, its.wN)) * its.wN;
              r.d = sample_diffuse(r.d, its, rng_state);

              throughput *= vec3(0.0, 0.7, 0.0);
            } else if(its.material == MBLUEDIFFUSE) {
              r.o = its.wP - 0.0001 * sign(dot(r.d, its.wN)) * its.wN;
              r.d = sample_diffuse(r.d, its, rng_state);

              throughput *= vec3(0.0,0.0,0.7);
            } else if(its.material == MGLOSSYREFLECTION) {
              r.o = its.wP - 0.0001 * sign(dot(r.d, its.wN)) * its.wN;
              r.d = sample_glossy(r.d, its, rng_state);
            } else if(its.material == MWEIRD) {            
                if(sample_next(rng_state) < 0.5)
                {
                  r.o = its.wP - 0.0001 * sign(dot(r.d, its.wN)) * its.wN;
                  r.d = sample_diffuse(r.d, its, rng_state);
                }
                else
                {
                  // Note the minus sign here!
                  r.o = its.wP - 0.0001 * its.wN;
                }
            }
            else {
              throughput = vec3(0);
              break;
            }
          }
        }
        else
        {
          // Ray hit the sky
          // throughput *= background(r_d);
          // break;

          uint idx_c = min(int(sample_next(rng_state) * 3), 2);

          float t = sample_Tr(sigmat, rng_state)[idx_c];

          vec3 x = r.o + t * r.d;	// Point in the media
          
          float Pac = sigmaa[idx_c] / sigmat[idx_c]; // Equivalent to 0
          float Psc = sigmas[idx_c] / sigmat[idx_c];

          float event_choice = sample_next(rng_state);

          // * ft(t) / p(t) = 1 when every spectre are equals
          vec3 tmp = (sigmat * Tr(sigmat, t));
          throughput *= 3.0*tmp / (tmp.x + tmp.y + tmp.z);

          if (event_choice < Pac){
            // Participing Media is Emitting No Light. Still, if we hit an absorption particles, we terminate the path.
            throughput = vec3(0.0);
            break;
          } else { // event_choice < [(Pac + Psc) == 1.0]
            vec3 nw = normalize(sample_fp_hg(g, r.d, rng_state));
            r.o = x;
            r.d = nw;

            // * (m_sigma_s/m_sigma_t)/Psc * fp(x, w, w') / p(w'|x, w)
            // fp(x, w, w') / p(w'|x, w) = 1 (We sample the direction from the same distribution as fp)
            // (m_sigma_s/m_sigma_t)/Psc = 1 when every spectre are equals
            throughput *= (sigmas/sigmat)/Psc;
          }         
        }
      }
    }
      Lo /= float(NUM_SAMPLES);
  }

  // Get the index of this invocation in the buffer:
  uint linearIndex       = resolution.x * pixel.y + pixel.x;
  imageData[linearIndex] = Lo;  // Take the average

}