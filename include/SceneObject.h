#ifndef SCENE_OBJECT_H
#define SCENE_OBJECT_H

#include "common.h"
#include <glm/fwd.hpp>
#include <glm/mat4x4.hpp>
#include <glm/gtx/transform.hpp>

using mat4 = glm::mat4x4;
using vec4 = glm::vec4;

class SceneObject {
  protected:
    mat4 model_     = mat4(1); // local transformation matrix
    mat4 model_inv_ = mat4(1); // local transformation matrix inverse

    vec3 translations_ = vec3(0); // cumulative vector of tranlations
    vec3 rotations_    = vec3(0);    // cumulative vector of rotations (in degrees)

    vec3 speed_ = vec3(0); // x,y,z movement done in one time tick
    vec3 spin_  = vec3(0);  // x,y,z deg of rotation done in one time tick

    // TOOD implement parent and has_parent?

    //__host__ __device__ void init() {
    //  model_         = mat4(1.0); // identity
    //  model_inv_ = mat4(1.0); // identity

    //  translations_ = vec3(0);
    //  rotations_    = vec3(0);

    //  speed_ = vec3(0);
    //  spin_  = vec3(0);
    //}
    __host__ __device__ void update_model();
    __host__ __device__ void update_model_inv();

  public:
    //__host__ __device__ SceneObject() { init(); }
    __host__ __device__ vec3   localToWorld (const vec3& target, const bool as_point) const;
    __host__ __device__ vec3   localToWorldV(const vec3& target) const;
    __device__ point3 localToWorldP(const point3& target) const;

    __host__ __device__ vec3   worldToLocal (const vec3& target, const bool as_point) const;
    __host__ __device__ vec3   worldToLocalV(const vec3& target) const;
    __host__ __device__ point3 worldToLocalP(const point3& target) const;

    __host__ __device__ SceneObject translate(const float x, const float y, const float z);
    __host__ __device__ SceneObject translate(const vec3& t);

    __host__ __device__ SceneObject rotate(const float deg_x, const float deg_y, const float deg_z);
    __host__ __device__ SceneObject rotate(const vec3& rotations);
    __host__ __device__ SceneObject rotateX(const float deg);
    __host__ __device__ SceneObject rotateY(const float deg);
    __host__ __device__ SceneObject rotateZ(const float deg);

    __host__ __device__ SceneObject set_speed(const vec3& speed);
    __host__ __device__ SceneObject set_spin(const vec3& spin);

    virtual __host__ __device__ void update();
};

#endif
