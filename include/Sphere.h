#ifndef SPHERE_H
#define SPHERE_H

#include "common.h"
#include "ImplicitShape.h"

class Sphere : public ImplicitShape {
  private:
    //point3 center_ = point3(0);
    float radius_ = 0.5;
    //color albedo_ = color(0.35);

    //static constexpr float gradient_delta_ = 10e-5; // delta used to compute gradient (normal)

    void init() {
      ImplicitShape::init();
    }
  public:
    Sphere(const float& radius) : radius_(radius) {
      init();
    }
    Sphere(const float& radius, const color& albedo) : radius_(radius) {
      init();
      cdiff_ = albedo;
    }
    Sphere(const point3& center, const float& radius, const color& albedo) : radius_(radius) {
      init();
      translate(center);
      cdiff_ = albedo;
      update();
    }
    //Sphere(const point3& center, const float& radius) : center_(center), radius_(radius) {}

    __device__ float getDist(const point3& point) const {
      //point3 p = point - center_; // very basic from world to local
      //return glm::length(p) - radius_;
      if (
          threadIdx.x + blockDim.x * blockIdx.x == 0 &&
          threadIdx.y + blockDim.y * blockIdx.y == 0
          ) {
        printf("\n\npoint = (%f,%f,%f)\n", point.x, point.y, point.z);
        point3 p = wordToLocalP(point);
        printf("p = (%f,%f,%f)\n", p.x, p.y, p.z);
        printf("model : [\n %f,%f,%f,%f,\n %f,%f,%f,%f,\n %f,%f,%f,%f,\n %f,%f,%f,%f\n ]\n",
            model_[0][0], model_[1][0], model_[2][0], model_[3][0],
            model_[0][1], model_[1][1], model_[2][1], model_[3][1],
            model_[0][2], model_[1][2], model_[2][2], model_[3][2],
            model_[0][3], model_[1][3], model_[2][3], model_[3][3]
            );
        printf("translations_ = (%f,%f,%f)\n", translations_.x, translations_.y, translations_.z);
      }
      return glm::length( wordToLocalP(point)) - radius_;
    }

    __device__ vec3 getNormalAt(const point3& p) const {
      return glm::normalize(vec3(
            getDist(
              p+vec3(gradient_delta_,0,0)) - getDist(p + vec3(-gradient_delta_,0,0)),
            getDist(
              p+vec3(0,gradient_delta_,0)) - getDist(p + vec3(0,-gradient_delta_,0)),
            getDist(
              p+vec3(0,0,gradient_delta_)) - getDist(p + vec3(0,0,-gradient_delta_))
            ));
    }
};

#endif
