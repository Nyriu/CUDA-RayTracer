#ifndef SCENEBUILDER_H
#define SCENEBUILDER_H

#include "common.h"
#include "ImplicitShape.h"
#include "Camera.h"
#include "Light.h"
#include "Scene.h"

#define rnd(x) (x*rand() / (float)RAND_MAX)



class SceneBuilder {
  private:
    float rand_seed_ = -1;

    int n_objs_   = 5;
    int n_lights_ = 1;


    BenchmarkTimeWriter *bTWriter_;

  public:
    enum PreBuiltScene { none, simple_moving };

    SceneBuilder(
        int n_objs = 5,
        int n_lights = 1
        ) :
      rand_seed_(-1),
      n_objs_(n_objs),
      n_lights_(n_lights) {}

    SceneBuilder(
        int n_objs,
        int n_lights,
        BenchmarkTimeWriter *bTWriter
        ) :
      rand_seed_(-1),
      n_objs_(n_objs),
      n_lights_(n_lights),
      bTWriter_(bTWriter) {}

    SceneBuilder(
        float rand_seed,
        int n_objs = 5,
        int n_lights = 1
        ) :
      rand_seed_(rand_seed),
      n_objs_(n_objs),
      n_lights_(n_lights) {}

    bool generate_scene(Scene *sce, Camera *cam) const;
    bool generate_scene(PreBuiltScene scene_idx, Scene *sce, Camera *cam) const;
  private:
    bool genSce_random(Scene *sce, Camera *cam) const;
    bool genSce_simple_moving(Scene *sce, Camera *cam) const;
};

#endif
