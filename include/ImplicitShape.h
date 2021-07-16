#ifndef IMPLICIT_SHAPE_H
#define IMPLICIT_SHAPE_H

#include "SceneObject.h"
#include <glm/common.hpp>
#include <glm/geometric.hpp>

enum ShapeType { none, sphere, cube, torus };
class ImplicitShapeInfo {
  public:
    ShapeType shape_type = ShapeType::none;

    color cdiff     = color(0.5);
    color cspec     = color(0.04);
    float shininess = 2;

    vec3 translations = vec3(0);
    vec3 rotations = vec3(0);

    vec3 speed = vec3(0);
    vec3 spin  = vec3(0);

    float additional_0 = 0.5;
    //float additional_1 = 0.25;

    __host__ __device__ ImplicitShapeInfo(
        const ShapeType& shape_type_i,

        const color& cdiff_i,
        const color& cspec_i,
        const float  shininess_i,

        const vec3& translations_i,
        const vec3& rotations_i,

        const vec3& speed_i,
        const vec3& spin_i
        ) :
        shape_type(shape_type_i), 
        cdiff(cdiff_i),
        cspec(cspec_i),
        shininess(shininess_i),
        translations(translations_i),
        rotations(rotations_i),
        speed(speed_i),
        spin(spin_i) {}
    __host__ __device__ ImplicitShapeInfo(
        const ShapeType& shape_type_i,

        const color& cdiff_i,
        const color& cspec_i,
        const float  shininess_i,

        const vec3& translations_i,
        const vec3& rotations_i,

        const vec3& speed_i,
        const vec3& spin_i,

        const float additional_i
        ) :
        ImplicitShapeInfo(
            shape_type_i,
            cdiff_i,
            cspec_i,
            shininess_i,
            translations_i,
            rotations_i,
            speed_i,
            spin_i)
        {
          if (shape_type_i == ShapeType::sphere) {
            additional_0 = additional_i;
          } else if (shape_type_i == ShapeType::cube) {
            additional_0 = additional_i;
          } else if (shape_type_i == ShapeType::torus) {
            additional_0 = additional_i;
          } else {
            printf("\n\nAAAAAAAAAAA that's BAD!!!!\n\n");
          }
        }
};


class ImplicitShape : public SceneObject {
  protected:
    color cdiff_     = color(0.5);
    color cspec_     = color(0.04);
    float shininess_ = 2;


    static constexpr float gradient_delta_ = 10e-5; // delta used to compute gradient (normal)

  public:
    __host__ __device__ ImplicitShape() = default;
    __host__ __device__ ImplicitShape(const color& albedo) : cdiff_(albedo) {}
    __host__ __device__ ImplicitShape(const color& albedo, const color& spec, float shininess) : 
      cdiff_(albedo), cspec_(spec), shininess_(shininess) {}
    __host__ __device__ ImplicitShape(const ImplicitShapeInfo& isi) :
      SceneObject(),
      cdiff_(isi.cdiff),
      cspec_(isi.cspec),
      shininess_(isi.shininess) {
        translations_ = isi.translations;
        rotations_ = isi.rotations;
        set_spin(isi.spin);
      }

    __device__ virtual float getDist(const point3& p) const {
      if (
          threadIdx.x + blockIdx.x * blockDim.x == 0 &&
          threadIdx.y + blockIdx.y * blockDim.y == 0) {
        printf("qui\n");
      }
      return 0;
    }

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

    __host__ __device__ virtual ImplicitShapeInfo get_info() const {
      return ImplicitShapeInfo(
          ShapeType::none,
          cdiff_, cspec_, shininess_,
          translations_, rotations_,
          speed_, spin_
          );
    }
};

class Sphere : public ImplicitShape {
  private:
    float radius_ = 0.5;

  public:
    Sphere(const float& radius) : radius_(radius) {}
    Sphere(const float& radius, const color& albedo) : ImplicitShape(albedo), radius_(radius) {}
    Sphere(const point3& center, const float& radius, const color& albedo) :
      ImplicitShape(albedo), radius_(radius) {
        translate(center);
      }
    __host__ __device__ Sphere(
        const point3& center, const float& radius,
        const color& albedo, const color& spec, float shininess) : 
      ImplicitShape(albedo, spec, shininess), radius_(radius) {
        translate(center);
      }
    __host__ __device__ Sphere(const ImplicitShapeInfo& isi) :
      ImplicitShape(isi), radius_(isi.additional_0) {}

    __device__ float getDist(const point3& point) const override {
      return glm::length( worldToLocalP(point)) - radius_;
    }

    __host__ __device__ ImplicitShapeInfo get_info() const override {
      return ImplicitShapeInfo(
          ShapeType::sphere,
          cdiff_, cspec_, shininess_,
          translations_, rotations_,
          speed_, spin_,
          radius_
          );
    }
};


class Cube : public ImplicitShape {
  private:
    float half_dim_ = .5;
  public:
    Cube(const float& half_dim) : half_dim_(half_dim) {}
    Cube(const point3& center, const float& half_dim, const color& albedo) :
      ImplicitShape(albedo), half_dim_(half_dim) {
        translate(center);
      }
    __host__ __device__ Cube(
        const point3& center, const float& half_dim,
        const color& albedo, const color& spec, float shininess) : 
      ImplicitShape(albedo, spec, shininess), half_dim_(half_dim) {
        translate(center);
      }
    __host__ __device__ Cube(const ImplicitShapeInfo& isi) :
      ImplicitShape(isi), half_dim_(isi.additional_0) {
      }

    __device__ float getDist(const point3& point) const override {
      point3 p = worldToLocalP(point);
      point3 q = glm::abs(p) - vec3(half_dim_);
      return glm::length(glm::max(q,vec3(0.0))) +
        min(glm::max(q.x, glm::max(q.y,q.z)), 0.0);
    }

    __host__ __device__ ImplicitShapeInfo get_info() const override {
      return ImplicitShapeInfo(
          ShapeType::cube,
          cdiff_, cspec_, shininess_,
          translations_, rotations_,
          speed_, spin_,
          half_dim_
          );
    }
};

class Torus : public ImplicitShape {
  private:
    float r0_=1; //, r1_=.2;
  public:
    Torus(const point3& center, const float& r0, const color& albedo) :
      ImplicitShape(albedo), r0_(r0) {
        translate(center);
      }
    __host__ __device__ Torus(
        const point3& center, const float& r0,
        const color& albedo, const color& spec, float shininess) : 
      ImplicitShape(albedo, spec, shininess), r0_(r0) {
        translate(center);
      }

    __host__ __device__ Torus(const ImplicitShapeInfo& isi) :
      ImplicitShape(isi), r0_(isi.additional_0) {}

    __device__ float getDist(const point3& point) const override {
      point3 p = worldToLocalP(point);
      // to 2D plane
      float tmpx = std::sqrt(p.x*p.x + p.z*p.z) - r0_;
      float tmpy = p.y;
      //return sqrtf(tmpx * tmpx + tmpy * tmpy) - r1_;
      return sqrtf(tmpx * tmpx + tmpy * tmpy) - .09;
    }

    __host__ __device__ ImplicitShapeInfo get_info() const override {
      return ImplicitShapeInfo(
          ShapeType::torus,
          cdiff_, cspec_, shininess_,
          translations_, rotations_,
          speed_, spin_,
          r0_
          );
    }
};

#endif

