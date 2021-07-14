#ifndef RENDERER_H
#define RENDERER_H

#include "Camera.h"
#include "Scene.h"
#include "Tracer.h"

class Renderer {
  private:
    Tracer *tracer_;
  public:
    __host__ void render(
        Camera *cam,
        Scene *sce,
        uchar4 *devPtr);
};

#endif
