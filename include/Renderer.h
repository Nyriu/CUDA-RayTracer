#ifndef RENDERER_H
#define RENDERER_H

#include "Camera.h"
#include "Scene.h"
#include "Tracer.h"

class Renderer {
  private:
    Tracer *tracer_;
    int current_tick_  = 0;
    int max_num_tick_  = -1; // unlimited

    Camera *devCamPtr_ = nullptr;
    Tracer *devTrcPtr_ = nullptr; // TODO
    Scene  *devScePtr_ = nullptr;

    bool done_cuda_free_ = false;


  public:
    __host__ Renderer() = default;

    __host__ Renderer(
        Camera *cam,
        Scene *sce,
        int max_num_tick = -1
        );

    __host__ void render(uchar4 *devPtr);
};

#endif
