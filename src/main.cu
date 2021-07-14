#define GL_GLEXT_PROTOTYPES

#include <iostream>

#include <GLFW/glfw3.h>

#include <cuda.h>
#include <cuda_gl_interop.h>

#define rnd(x) (x*rand() / (float)RAND_MAX)

#include "common.h"
#include "ImplicitShape.h"
#include "Light.h"
#include "Scene.h"
#include "Ray.h"
#include "Camera.h"
#include "Tracer.h"
#include "Renderer.h"


static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
  // Press ESC to close the window
  if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
    glfwSetWindowShouldClose(window, GLFW_TRUE);
}


GLuint bufferObj;
cudaGraphicsResource *resource;

void device_setup(
    // input
    // output
	  GLFWwindow **window,
    uchar4 **devPtr
    ) {
  cudaDeviceProp prop;
  int dev;

  memset(&prop, 0, sizeof(cudaDeviceProp));
  prop.major = 1;
  prop.minor = 0;
  HANDLE_ERROR(
      cudaChooseDevice(&dev,&prop)
      );

	if (!glfwInit()) exit(EXIT_FAILURE);
	if (atexit(glfwTerminate)) {
		glfwTerminate();
		exit(EXIT_FAILURE);
	}

	*window = glfwCreateWindow(IMG_W, IMG_H, "GLFW Window", NULL, NULL);
	if (!window) exit(EXIT_FAILURE);
  glfwSetKeyCallback(*window, key_callback);

	glfwMakeContextCurrent(*window);
	glfwSwapInterval(1);

  // TODO ARB or not ARB
	//glGenBuffers(1, &pbo);
	//glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
	//glBufferData(GL_PIXEL_UNPACK_BUFFER, 4 * sizeof(GLubyte)*WIDTH*HEIGHT, NULL, GL_DYNAMIC_DRAW);
  glGenBuffers(1, &bufferObj);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, bufferObj);
  glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, IMG_H*IMG_W*4, NULL, GL_DYNAMIC_DRAW_ARB);

  HANDLE_ERROR(
      cudaGraphicsGLRegisterBuffer(
        &resource,
        bufferObj,
        cudaGraphicsMapFlagsNone
        )
      );
  size_t size;
  HANDLE_ERROR(
      cudaGraphicsMapResources(1, &resource, NULL)
      );
  HANDLE_ERROR(
      cudaGraphicsResourceGetMappedPointer(
        (void**)devPtr,
        &size,
        resource
        )
      );
}

void device_terminate() {
  // Terminate
  HANDLE_ERROR(
      cudaGraphicsUnmapResources(1, &resource, NULL)
      );
  HANDLE_ERROR(
      cudaGraphicsUnregisterResource(resource)
      );
  glBindBuffer( GL_PIXEL_UNPACK_BUFFER_ARB, 0);
  glDeleteBuffers(1, &bufferObj);

  glfwTerminate();
}


void scene_setup(
    // input
    // output
    Scene& sce
    ) {
  // Init Random scene
  int n_obj = 100;
  srand( (unsigned)time(NULL) );
  //srand( (unsigned) 12345 );
  for (int i=0; i<n_obj; i++) {
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

    //printf("alb  = (%f,%f,%f)\n", alb.x, alb.y, alb.z);
    //printf("spec = (%f,%f,%f)\n", spec.x, spec.y, spec.z);
    //printf("shin = %f\n", shininess);

    float shape_prob = (float) rnd(1.0f);
    if (shape_prob < 0.33) {
      float radius = (float) rnd(0.3f) + 0.1;
      sce.addShape(new Sphere(
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
      sce.addShape(obj);
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
      sce.addShape(obj);
    }
  }
  sce.addLight(new PointLight(point3(5,4,3), color(1), 80));
  sce.addAmbientLight(new AmbientLight());
}



int main() {
	GLFWwindow* window;
  uchar4* devPtr;

  device_setup(&window, &devPtr);

  Camera cam;
  Scene sce;

  scene_setup(sce);

  { // manual scene setup
    //sce.addShape(new Sphere(1, color(0.5, 0.8, 0.7)));
    //sce.addShape(new Sphere(point3(1,0,0), 1, color(0.5, 0.8, 0.7)));

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

    //sce.addLight(new PointLight(point3(5,4,3), color(1), 80));
    //sce.addLight(new PointLight(point3(5,4,3), color(1)));
    //sce.addLight(new Light(point3(5,4,3), color(1)));
    //sce.addLight(new PointLight(point3( 5,4,3), color(0.3,1,0.5)));
    //sce.addLight(new PointLight(point3(-4,4,3), color(1,0.3,0.5), 50));
    //sce.addAmbientLight(new AmbientLight());
  }

  Renderer renderer;
  renderer.render(&cam, &sce, devPtr);

	while (!glfwWindowShouldClose(window)) {
		glDrawPixels(IMG_W, IMG_H, GL_RGBA, GL_UNSIGNED_BYTE, 0);
		glfwSwapBuffers(window);
    // Poll and process events
    glfwPollEvents();
    // Here update scene to be interactive // TODO
	}

  device_terminate();

  return 0;
}

