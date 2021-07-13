#ifndef SCENE_H
#define SCENE_H

#include <vector>

#include "common.h"
#include "ImplicitShape.h"
#include "Camera.h"
#include "Light.h"

class Scene {
  public:
    using Shapes = std::vector<ImplicitShape*>; 
    using Lights = std::vector<Light*>;

  private:
    Shapes shapes_;
    Lights lights_;
    AmbientLight* ambientLight_ = nullptr;

    Scene *devPtr_ = nullptr;
    //ImplicitShape *devShapes_ = nullptr;
    int shapes_num_ = 0; // number of shapes
    Light *devLights_ = nullptr;
    int lights_num_ = 0; // number of lights
    AmbientLight *devAmbLight_ = nullptr;
  public:
    Scene() = default;
    Scene(Shapes shapes, Lights lights) : shapes_(shapes), lights_(lights) {}

    void addShape(ImplicitShape* shape);
    void addLight(Light* light);
    void addAmbientLight(AmbientLight* light);

    __device__ __host__ bool hasAmbientLight() const { return ambientLight_ != nullptr; }

    __device__ ImplicitShape* getShapes() const; // { return devShapes_; }
    __device__ int getShapesNum() const { return shapes_num_; }
    __device__ Light* getLights() const { return devLights_; }
    __device__ int getLightsNum() const { return lights_num_; }
    __device__ AmbientLight* getAmbientLight() const { return devAmbLight_; }

  private:
    __host__ void shapes_to_device();
    __host__ void lights_to_device();
  public:
    /** moves Scene's data to device and returns the device pointer to scene **/
    __host__ Scene* to_device();
};

#endif
