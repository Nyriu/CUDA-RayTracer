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
    //enum PreBuiltScene { none, simple_moving, empty, easy_0, easy_1, medium_0, medium_1, hard_0, hard_1, impossible };
    enum PreBuiltScene { none, simple_moving, empty, easy_0, easy_1,             medium_1};

    SceneBuilder(
        int n_objs = 5,
        int n_lights = 1
        ) :
      rand_seed_(-1),
      n_objs_(n_objs),
      n_lights_(n_lights) {
        bTWriter_ = new BenchmarkTimeWriter;
      }


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
      n_lights_(n_lights) {
        bTWriter_ = new BenchmarkTimeWriter;
      }

    bool generate_scene(Scene *sce, Camera *cam) const;
    bool generate_scene(PreBuiltScene scene_idx, Scene *sce, Camera *cam) const;
  private:
    bool genSce_random(Scene *sce, Camera *cam) const;
    bool genSce_simple_moving(Scene *sce, Camera *cam) const;

    // benchmarking scenes
    bool genSce_empty(Scene *sce, Camera *cam) const;

    bool genSce_easy_0(Scene *sce, Camera *cam) const;
    bool genSce_easy_1(Scene *sce, Camera *cam) const;

    bool genSce_medium_0(Scene *sce, Camera *cam) const;
    bool genSce_medium_1(Scene *sce, Camera *cam) const;

    bool genSce_hard_0(Scene *sce, Camera *cam) const;
    bool genSce_hard_1(Scene *sce, Camera *cam) const;

    bool genSce_impossible(Scene *sce, Camera *cam) const;
};

#endif
