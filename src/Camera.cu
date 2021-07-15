#include "Camera.h"
#include <glm/ext/matrix_transform.hpp>
#include <glm/ext/quaternion_geometric.hpp>
#include <glm/fwd.hpp>
#include <glm/geometric.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/matrix.hpp>

__device__ Ray Camera::generate_ray(float u, float v) const { // input NDC Coords
  // Put coords in [-1,1] // (-1,-1) is bottom-left
  float su = u * 2 - 1; // Screen Coord
  float sv = v * 2 - 1; // Screen Coord

  // Aspect Ratio
  su *= aspect_; // x in [-asp ratio, asp ratio]
  // y in [-1,1] (as before)

  // Field Of View
  su *= std::tan(fov_/2);
  sv *= std::tan(fov_/2);

  //float scale = 1;
  //// Scale
  //su *= scale;
  //sv *= scale;

  // From ScreenCoords to WorldCoords
  //point3 center = translations_;
  //point3 p    = point3(su,sv,center.z - 1) ;
  //point3 orig = center;

  ////point3 p = localToWorldP(point3(su,sv,-1));
  //point3 p = localToWorldP(point3(su,sv,
  //      dir_.z < 0 ? -1 : 1 // TODO
  //      //dir_.z < 0 ? 1 : -1 // TODO
  //      //-1
  //      ));
  //point3 orig = translations_;
  //vec3 dir = glm::normalize(p - orig);
  ////dir = glm::normalize(vec3(dir.x,dir.y,-1));

  // From ScreenCoords to WorldCoords
  point3 p = point3(su,sv,-1);

  // TODO??
  //creare view matrix con cui ruotare la dir e applicare il flip della Z

  point3 orig = localToWorldP(point3(0));
  //point3 orig = translations_;
  vec3 dir = glm::normalize(localToWorldP(p) - orig);
  //dir = vec3(dir.x,dir.y,-dir.z);

  //if ( false && (
  //if ((
  //    threadIdx.x + blockDim.x * blockIdx.x +
  //    threadIdx.y + blockDim.y * blockIdx.y +
  //    threadIdx.z + blockDim.z * blockIdx.z == 0)) {
  //  printf("p    = (%f,%f,%f)\n",    p.x,    p.y,    p.z);
  //  printf("orig = (%f,%f,%f)\n", orig.x, orig.y, orig.z);
  //  printf("dir  = (%f,%f,%f)\n",  dir.x,  dir.y,  dir.z);

  //  printf("\n");
  //  printf("dir_ = (%f,%f,%f)\n",  dir_.x,  dir_.y,  dir_.z);
  //  printf("model : [\n %f,%f,%f,%f,\n %f,%f,%f,%f,\n %f,%f,%f,%f,\n %f,%f,%f,%f\n ]\n",
  //      model_[0][0], model_[1][0], model_[2][0], model_[3][0],
  //      model_[0][1], model_[1][1], model_[2][1], model_[3][1],
  //      model_[0][2], model_[1][2], model_[2][2], model_[3][2],
  //      model_[0][3], model_[1][3], model_[2][3], model_[3][3]
  //      );
  //}
  return Ray(orig, dir);
}

__host__ __device__ mat3 align_by_rotation(const vec3& a, const  vec3& b) {
  //TODO
  /** return rotation mat that rotates a onto b BOTH MUST BE UNIT VECTORS **/
  //if (glm::length(a) != 1.0 || glm::length(b) != 1.0) {
  //  printf("glm::length(a) = %f\n", glm::length(a));
  //  printf("glm::length(b) = %f\n", glm::length(b));
  //  printf("\nalign_by_rotation : ERROR not unit vectors!!\n");
  //  exit(1);
  //}
  //vec3 v = glm::cross(a,b); 
  float s = glm::length(glm::cross(a,b)); // sine of angle
  float c = glm::dot(a,b);  // cosine of angle
  if (c == -1) {
    printf("\nalign_by_rotation : ERROR cosine = -1 !!\n");
    exit(1);
  }
  //mat3 vx(
  //     0.f, -v.z,  v.y,
  //     v.z,  0.f, -v.x,
  //    -v.y,  v.x,  0.f
  //    );
  //mat3 r = mat3(
  //    1,0,0,
  //    0,1,0,
  //    0,0,1
  //    //) + vx + vx*vx * (1-c)/(s*s); // can be simplified into
  //    ) + vx + vx*vx * (1/(1+c));

  mat3 G = mat3(
      c,-s,0,
      s, c,0,
      0, 0,1
      );

  vec3 u = a; // norm vect projection of b onto a
  vec3 v = (b-glm::dot(a,b)*a)/glm::length((b-glm::dot(a,b)*a)); // norm vector rejection of b onto a
  vec3 w = glm::cross(b,a);

  mat3 F( u,v,w );

  //printf("u = (%f,%f,%f)\n",  u.x,  u.y,  u.z);
  //printf("v = (%f,%f,%f)\n",  v.x,  v.y,  v.z);
  //printf("w = (%f,%f,%f)\n",  w.x,  w.y,  w.z);

  //printf("F : [\n %f,%f,%f,\n %f,%f,%f,\n %f,%f,%f\n ]\n",
  //    F[0][0], F[1][0], F[2][0],
  //    F[0][1], F[1][1], F[2][1],
  //    F[0][2], F[1][2], F[2][2]
  //    );

  mat3 U = glm::inverse(F) * G * F;
  return U;
}

__host__ __device__ Camera Camera::look_at(const point3& p) {
  //vec3 dir = p - eye; // TODO
  //vec3 dir = glm::normalize(p - translations_);
  //float angle = 
  //  glm::acos(
  //      glm::dot(dir_, dir) / (
  //        glm::length(dir_)*glm::length(dir)
  //        )
  //      );

  //printf("p = (%f,%f,%f)\n",  p.x,  p.y,  p.z);
  //printf("dir_ = (%f,%f,%f)\n",  dir_.x,  dir_.y,  dir_.z);
  //printf("dir = (%f,%f,%f)\n",  dir.x,  dir.y,  dir.z);
  //printf("angle = %f\n", angle);

  //glm::rotate(

  //vec3 a = glm::normalize(vec3(0,0,1));
  //vec3 b = glm::normalize(vec3(1,0,0));
  //mat3 r = align_by_rotation(a,b);
  //vec3 c = r*r*a;


  //printf("a = (%f,%f,%f)\n",  a.x,  a.y,  a.z);
  //printf("b = (%f,%f,%f)\n",  b.x,  b.y,  b.z);

  //printf("r : [\n %f,%f,%f,\n %f,%f,%f,\n %f,%f,%f\n ]\n",
  //    r[0][0], r[1][0], r[2][0],
  //    r[0][1], r[1][1], r[2][1],
  //    r[0][2], r[1][2], r[2][2]
  //    );

  //printf("c = (%f,%f,%f)\n",  c.x,  c.y,  c.z);

  printf("\nCamera::look_at NOT IMPLEMENTED!\n");
  return *this;
}

