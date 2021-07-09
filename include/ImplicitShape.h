#ifndef IMPLICIT_SHAPE_H
#define IMPLICIT_SHAPE_H

#include "SceneObject.h"

class ImplicitShape : public SceneObject {
  protected:
    color cdiff_ = color(0.5);
    color cspec_ = color(0.5);
    float shininess_ = 2;

    static constexpr float gradient_delta_ = 10e-5; // delta used to compute gradient (normal)

    __host__ __device__ void init() {
      SceneObject::init();
      cdiff_ = color(0.5);
      cspec_ = color(0.5);
      shininess_ = 2;
    }
  public:
    __device__ float getDist(const point3& p) const { return 0; }

    __host__ __device__ ImplicitShape setAlbedo(const color& color) {
      cdiff_ = color;
      return *this;
    }
    __host__ __device__ ImplicitShape setSpecular(const color& color) {
      cspec_ = color;
      return *this;
    }
    __host__ __device__ ImplicitShape setShininess(float shininess) {
      shininess_ = shininess;
      return *this;
    }

    __host__ __device__ color getAlbedo()    const { return cdiff_; }
    __host__ __device__ color getSpecular()  const { return cspec_; }
    __host__ __device__ float getShininess() const { return shininess_; }

    __host__ __device__ color getAlbedo   (const point3& p) const { return getAlbedo(); }
    __host__ __device__ color getSpecular (const point3& p) const { return getSpecular(); }
    __host__ __device__ float getShininess(const point3& p) const { return getShininess(); }

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
