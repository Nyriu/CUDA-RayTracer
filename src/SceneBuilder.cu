#include "SceneBuilder.h"


bool SceneBuilder::generate_scene(
    Scene *sce, Camera *cam,
    PreBuiltScene scene_idx
    ) const {
  if (rand_seed_ < 0) {
    srand( (unsigned)time(NULL) );
  } else {
    srand( (unsigned) rand_seed_ );
  }

  if (scene_idx == PreBuiltScene::none) {
    return genSce_random(sce, cam);
  }
  if (scene_idx == PreBuiltScene::simple_moving) {
    return genSce_simple_moving(sce, cam);
  }
  // TODO add other prebuilt cases
  return false;
}

bool SceneBuilder::genSce_random(Scene *sce, Camera *cam) const {
  // TODO add parameters to pass
  for (int i=0; i<n_objs_; i++) {
    point3 pos(
        (float) rnd(4.0f) - 2,
        (float) rnd(4.0f) - 2,
        (float) rnd(4.0f) - 2
        );
    color alb(
        (float) rnd(1.0f),
        (float) rnd(1.0f),
        (float) rnd(1.0f));
    //float min_spec = 0.04f;
    //float max_spec = 0.2f;
    color spec(
        //(float) rnd(min_spec + max_spec) - (min_spec+max_spec)/2,
        //(float) rnd(min_spec + max_spec) - (min_spec+max_spec)/2,
        //(float) rnd(min_spec + max_spec) - (min_spec+max_spec)/2);
      0.04);
    //0.2);
    //.913,.922,.924);
    float shininess = 2; // 70; // (float) rnd(60.0f);

    vec3 spin(
        (float) rnd(2.0f)-rnd(2.0f),
        (float) rnd(2.0f)-rnd(2.0f),
        (float) rnd(2.0f)-rnd(2.0f));

    //printf("alb  = (%f,%f,%f)\n", alb.x, alb.y, alb.z);
    //printf("spec = (%f,%f,%f)\n", spec.x, spec.y, spec.z);
    //printf("shin = %f\n", shininess);

    float shape_prob = (float) rnd(1.0f);
    if (shape_prob < 0.33) {
      float radius = (float) rnd(0.3f) + 0.1;
      sce->addShape(new Sphere(
            pos,
            radius,
            alb,
            spec,
            shininess
            )
          );
    } else if (shape_prob < 0.66) {
      float half_dim = (float) rnd(0.3f) + 0.1;
      auto obj = new Cube(
          pos,
          half_dim,
          alb,
          spec,
          shininess
          );
      obj->rotate(
          (float) rnd(90.0f),
          (float) rnd(90.0f),
          (float) rnd(90.0f)
          );
      obj->set_spin(spin);
      sce->addShape(obj);
    } else {
      float radius = (float) rnd(0.3f) + 0.1;
      auto obj = new Torus(
          pos,
          radius,
          alb,
          spec,
          shininess
          );
      obj->rotate(
          (float) rnd(90.0f),
          (float) rnd(90.0f),
          (float) rnd(90.0f)
          );
      obj->set_spin(spin);
      sce->addShape(obj);
    }
  }
  sce->addLight(new PointLight(point3(5,4,3), color(1), 80));
  sce->addAmbientLight(new AmbientLight());

  cam->move_to(point3(.5,.5, 5));
  cam->update();

  return true;
}



bool SceneBuilder::genSce_simple_moving(Scene *sce, Camera *cam) const {
  ImplicitShape *obj = nullptr;

  sce->addShape(new Sphere(.7, color(0.8, 0.5, 0.5)));
  sce->addShape(new Sphere(point3(1,0,0), .5, color(0.5, 0.8, 0.5)));
  {
    obj = new Cube(point3(0,1,0), .3, color(0.5, 0.8, 0.7));
    obj->set_spin(vec3(0,1,0));
    sce->addShape(obj);
  }
  {
    obj = new Cube(point3(0,0,.5), .35, color(0.5, 0.5, 0.8));
    obj->set_spin(vec3(1,1,1));
    sce->addShape(obj);
  }
  {
    obj = new Torus(point3(.5,.7,-1), .25, color(0.8, 0.2, 0.8));
    obj->set_spin(vec3(1,0,1));
    sce->addShape(obj);
  }

  //auto obj = Sphere(point3(1,0,0), 1, color(0.5, 0.8, 0.7));
  //auto obj = Sphere(1, color(0.7, 0.7, 0.7));
  //auto obj = Cube(1);
  //auto obj = Torus(point3(1,0,0), 1, color(0.5, 0.8, 0.7));
  ////obj.translate(vec3(0,1,-1));
  //obj.translate(vec3(0,0,-1));
  //obj.translate(vec3(0,0,-1));
  //obj.translate(vec3(-1,1,1));
  //obj.rotate(vec3(45));
  //obj.update();
  //sce.addShape(&obj);

  //sce.addShape(new Sphere(point3(.5), .5, color(1,0,0)));

  sce->addLight(new PointLight(point3(5,4,3), color(1), 350));
  //sce->addLight(new PointLight(point3(5,4,3), color(1), 300));
  //sce.addLight(new PointLight(point3(5,4,3), color(1)));
  //sce.addLight(new Light(point3(5,4,3), color(1)));
  //sce.addLight(new PointLight(point3( 5,4,3), color(0.3,1,0.5)));
  //sce.addLight(new PointLight(point3(-4,4,3), color(1,0.3,0.5), 50));
  sce->addAmbientLight(new AmbientLight());

  //cam.move_to(point3(0,0,3));
  cam->move_to(point3(1.5,1.5,3));
  //cam.translate(2,3,3);
  cam->rotateX(-25);
  cam->rotateY(20);
  //cam.rotateZ(0.5);
  //cam.look_at(point3(1,1,1));
  cam->update();
  //cam.move_to(point3(0,0,3));
  //cam.translate(0,0,-1);
  //cam.update();
  return true;
}
