#ifndef IMPLICIT_SHAPE_H
#define IMPLICIT_SHAPE_H

#include "SceneObject.h"
#include <glm/common.hpp>
#include <glm/geometric.hpp>

enum ShapeType { none, sphere, cube };
class ImplicitShapeInfo {
  public:
    ShapeType shape_type = ShapeType::none;

    color cdiff = color(0.5);
    color cspec = color(0.5);
    float shininess = 2;

    vec3 translations = vec3(0);
    vec3 rotations = vec3(0);

    // Sphere stuff // TODO little ShapeInfo hierarchy here?
    float radius = 0.5;

    // Cube stuff // TODO little ShapeInfo hierarchy here?
    //vec3 half_dims = vec3(.5);
    float half_dim = .5;

    __host__ __device__ ImplicitShapeInfo(
        const ShapeType& shape_type_i,

        const color& cdiff_i,
        const color& cspec_i,
        const float  shininess_i,

        const vec3& translations_i,
        const vec3& rotations_i
        ) :
        shape_type(shape_type_i), 
        cdiff(cdiff_i),
        cspec(cspec_i),
        shininess(shininess_i),
        translations(translations_i),
        rotations(rotations_i) {}
    __host__ __device__ ImplicitShapeInfo(
        const ShapeType& shape_type_i,

        const color& cdiff_i,
        const color& cspec_i,
        const float  shininess_i,

        const vec3& translations_i,
        const vec3& rotations_i,

        const float additional_i
        ) :
        shape_type(shape_type_i), 
        cdiff(cdiff_i),
        cspec(cspec_i),
        shininess(shininess_i),
        translations(translations_i),
        rotations(rotations_i)
        {
          if (shape_type_i == ShapeType::sphere) {
            radius = additional_i;
          } else if (shape_type_i == ShapeType::cube) {
            half_dim = additional_i;
          } else {
            printf("\n\nAAAAAAAAAAA that's BAD!!!!\n\n");
          }
        }
};


class ImplicitShape : public SceneObject {
  protected:
    color cdiff_ = color(0.5);
    color cspec_ = color(0.5);
    float shininess_ = 2;

    static constexpr float gradient_delta_ = 10e-5; // delta used to compute gradient (normal)

  public:
    __host__ __device__ ImplicitShape() = default;
    __host__ __device__ ImplicitShape(const ImplicitShapeInfo& isi) :
      SceneObject(),
      cdiff_(isi.cdiff),
      cspec_(isi.cspec),
      shininess_(isi.shininess) {
        translate(isi.translations);
        rotate(isi.rotations);
        update();
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
          translations_, rotations_
          );
    }
};

class Sphere : public ImplicitShape {
  private:
    //point3 center_ = point3(0);
    float radius_ = 0.5;
    //color albedo_ = color(0.35);

    //static constexpr float gradient_delta_ = 10e-5; // delta used to compute gradient (normal)

  public:
    Sphere(const float& radius) : radius_(radius) {}
    Sphere(const float& radius, const color& albedo) : radius_(radius) {
      cdiff_ = albedo;
    }
    Sphere(const point3& center, const float& radius, const color& albedo) : radius_(radius) {
      translate(center);
      cdiff_ = albedo;
      update();
    }
    //Sphere(const point3& center, const float& radius) : center_(center), radius_(radius) {}
    __host__ __device__ Sphere(const ImplicitShapeInfo& isi) :
      ImplicitShape(isi), radius_(isi.radius) {}

    __device__ float getDist(const point3& point) const override {
      //point3 p = point - center_; // very basic from world to local
      //return glm::length(p) - radius_;
      //printf("I've been called\n");
      //if (
      //    threadIdx.x + blockDim.x * blockIdx.x == 0 &&
      //    threadIdx.y + blockDim.y * blockIdx.y == 0
      //    ) {
      //  printf("point = (%f,%f,%f)\n", point.x, point.y, point.z);
      //  printf("radius = %f\n", radius_);
      //  point3 p = worldToLocalP(point);
      //  printf("p = (%f,%f,%f)\n", p.x, p.y, p.z);
      //  printf("model : [\n %f,%f,%f,%f,\n %f,%f,%f,%f,\n %f,%f,%f,%f,\n %f,%f,%f,%f\n ]\n",
      //      model_[0][0], model_[1][0], model_[2][0], model_[3][0],
      //      model_[0][1], model_[1][1], model_[2][1], model_[3][1],
      //      model_[0][2], model_[1][2], model_[2][2], model_[3][2],
      //      model_[0][3], model_[1][3], model_[2][3], model_[3][3]
      //      );
      //  printf("translations_ = (%f,%f,%f)\n", translations_.x, translations_.y, translations_.z);
      //}
      return glm::length( worldToLocalP(point)) - radius_;
    }

    __host__ __device__ ImplicitShapeInfo get_info() const override {
      return ImplicitShapeInfo(
          ShapeType::sphere,
          cdiff_, cspec_, shininess_,
          translations_, rotations_,
          radius_
          );
    }
};


class Cube : public ImplicitShape {
  private:
    float half_dim_ = .5;
  public:
    Cube(const float& half_dim) : half_dim_(half_dim) {}
    Cube(const point3& center, const float& half_dim, const color& albedo) : half_dim_(half_dim) {
      translate(center);
      cdiff_ = albedo;
      update();
    }
    __host__ __device__ Cube(const ImplicitShapeInfo& isi) :
      ImplicitShape(isi), half_dim_(isi.half_dim) {}

    __device__ float getDist(const point3& point) const override {
      point3 p = worldToLocalP(point);
      point3 abs_p = glm::abs(p);

      vec3 v = abs_p - vec3(half_dim_);
      float d = glm::length(
          glm::max(v, vec3(0))
            );
          //vec3(
          //  max(v.x,.0f),
          //  max(v.y,.0f),
          //  max(v.z,.0f)
          //  ));
      return d;
    }

    __host__ __device__ ImplicitShapeInfo get_info() const override {
      return ImplicitShapeInfo(
          ShapeType::cube,
          cdiff_, cspec_, shininess_,
          translations_, rotations_,
          half_dim_
          );
    }
};



#endif

