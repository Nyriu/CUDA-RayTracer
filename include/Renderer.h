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

    bool verbose_ = true;
    bool benchmarking_ = false;

    BenchmarkTimeWriter *bTWriter_;
  public:
    __host__ Renderer() = default;

    __host__ Renderer(
        Camera *cam,
        Scene *sce,
        int max_num_tick = -1
        );

    __host__ void verbose(bool b) { verbose_ = b; };
    __host__ void benchmarking(bool b, BenchmarkTimeWriter *bTWriter=nullptr) {
      benchmarking_ = b;
      if (benchmarking_) {
        bTWriter_ = bTWriter;
      }
      if (bTWriter == nullptr) {
        std::cout << "ERROR : Renderer::benchmarking bad initialization!" << std::endl;
      }
    };
    __host__ void render(uchar4 *devPtr);
};

#endif
