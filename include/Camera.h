#ifndef CAMERA_H
#define CAMERA_H

#include "common.h"
#include "SceneObject.h"
#include "Ray.h"

class Camera : public SceneObject {
  private:
    vec3 dir_;

    float aspect_;
    float fov_;

  public:
    Camera() : SceneObject(point3(0,0,5)) {
      dir_ = glm::normalize(vec3(0,0,-1));
      aspect_ = 1;
      fov_ = 45;
    }
    Camera(const point3& center) : SceneObject(center) {
      dir_ = glm::normalize(vec3(0,0,-1));
      aspect_ = 1;
      fov_ = 45;
    }

    __device__ Ray generate_ray(float u, float v) const; // input NDC Coords

    __host__ __device__ Camera look_at(const point3& p);
};

#endif
