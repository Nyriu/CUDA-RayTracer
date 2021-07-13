#include "SceneObject.h"
#include <cmath>
#include <glm/ext/matrix_transform.hpp>
#include <glm/fwd.hpp>
#include <glm/matrix.hpp>

__host__ __device__ vec3 SceneObject::localToWorld(const vec3& target, const bool as_point) const {
  return model_ * vec4(target, (int) as_point);
}
__host__ __device__ vec3 SceneObject::localToWorldV(const vec3& target) const {
  return model_ * vec4(target,0);
}
__device__ point3 SceneObject::localToWorldP(const point3& target) const {

  if (threadIdx.x + blockDim.x * blockIdx.x == 0) {
    printf("hola\n");
  }
  return model_ * vec4(target,1);
}

__host__ __device__ vec3   SceneObject::worldToLocal (const vec3& target, const bool as_point) const {
  return model_inv_ * vec4(target, (int) as_point);
}
__host__ __device__ vec3   SceneObject::worldToLocalV(const vec3& target) const {
  return model_inv_ * vec4(target, 0);
}
__host__ __device__ point3 SceneObject::worldToLocalP(const point3& target) const {
  return model_inv_ * vec4(target, 1);
}


__host__ __device__ SceneObject SceneObject::translate(const float x, const float y, const float z) {
  return translate(vec3(x,y,z));
}
__host__ __device__ SceneObject SceneObject::translate(const vec3& t) {
  translations_ += t;
  return *this;
}

__host__ __device__ SceneObject SceneObject::rotate(const float deg_x, const float deg_y, const float deg_z) {
  rotateX(deg_x);
  rotateY(deg_y);
  rotateZ(deg_z);
  return *this;
}
__host__ __device__ SceneObject SceneObject::rotate(const vec3& rotations) {
  return rotate(
      rotations.x,
      rotations.y,
      rotations.z
      );
}
__host__ __device__ SceneObject SceneObject::rotateX(const float deg) {
  rotations_.x += deg;
  if (rotations_.x >= 360)
    rotations_.x -= 360;
  return *this;
}
__host__ __device__ SceneObject SceneObject::rotateY(const float deg) {
  rotations_.y += deg;
  if (rotations_.y >= 360)
    rotations_.y -= 360;
  return *this;
}
__host__ __device__ SceneObject SceneObject::rotateZ(const float deg) {
  rotations_.z += deg;
  if (rotations_.z >= 360)
    rotations_.z -= 360;
  return *this;
}

__host__ __device__ SceneObject SceneObject::set_speed(const vec3& speed) {
  speed_ = speed;
  return *this;
}
__host__ __device__ SceneObject SceneObject::set_spin(const vec3& spin) {
  spin_ = spin;
  return *this;
}

__host__ __device__ void SceneObject::update() {
  translate(speed_);
  rotate(spin_);
  update_model();
  update_model_inv();
}

__host__ __device__ void SceneObject::update_model() {
  model_ = glm::translate(model_,translations_);

  model_ = glm::rotate(model_, rotations_.x, vec3(1,0,0));
  model_ = glm::rotate(model_, rotations_.y, vec3(0,1,0));
  model_ = glm::rotate(model_, rotations_.z, vec3(0,0,1));
}

__host__ __device__ void SceneObject::update_model_inv() {
  //TODO can be speeded up?
  // look at GLM implementation vs CPU RayTracer
  // because this is a particular inverse case
  // look at Real-Time Rendering 4th pag 66
  // just use -translations and -rotations
  model_inv_ = glm::inverse(model_);
}

