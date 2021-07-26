#include <ctime>
#include <glm/geometric.hpp>
#include <iterator>
#include <ostream>
#define GL_GLEXT_PROTOTYPES

#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>

#include <chrono>

#include <GLFW/glfw3.h>

#include <cuda.h>
#include <cuda_gl_interop.h>

#include "common.h"
#include "Renderer.h"
#include "SceneBuilder.h"


bool NEXT = false;
static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
  // Press ESC to close the window
  if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
    glfwSetWindowShouldClose(window, GLFW_TRUE);
  if (
      (key == GLFW_KEY_L && action == GLFW_PRESS) ||
      (key == GLFW_KEY_RIGHT && action == GLFW_PRESS)
     ) NEXT = true;
}

GLuint bufferObj;
cudaGraphicsResource *resource;

void device_setup(GLFWwindow **window) {
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

  //glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE); // Hide GLFW window

	*window = glfwCreateWindow(IMG_W, IMG_H, "GLFW Window", NULL, NULL);
	if (!window) exit(EXIT_FAILURE);
  glfwSetKeyCallback(*window, key_callback);

	glfwMakeContextCurrent(*window);
	glfwSwapInterval(1);

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
}

void map_resource(uchar4 **devPtr) {
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

void unmap_resource() {
  HANDLE_ERROR(
      cudaGraphicsUnmapResources(1, &resource, NULL)
      );
}

void device_terminate() {
  HANDLE_ERROR(
      cudaGraphicsUnregisterResource(resource)
      );
  glBindBuffer( GL_PIXEL_UNPACK_BUFFER_ARB, 0);
  glDeleteBuffers(1, &bufferObj);

  glfwTerminate();
}


struct AdditionalInputs {
  int n_objs   = 0;
  int n_lights = 0;
};

void init_rnd(
    Camera *cam, Scene  *sce,
    AdditionalInputs *addin=nullptr) {
  int n_objs   = 5;
  int n_lights = 1;

  if (addin != nullptr) {
    n_objs = addin->n_objs;
    n_lights = addin->n_lights;
  }

  std::cout << "here" << std::endl;

  SceneBuilder sceBui(n_objs, n_lights);
  std::cout << "maybe here" << std::endl;
  sceBui.generate_scene(
      SceneBuilder::PreBuiltScene::none,
      sce, cam
      );
  std::cout << "but not here" << std::endl;
}

void init_easy_0(
    Camera *cam, Scene  *sce,
    AdditionalInputs *addin=nullptr) {
  if (addin != nullptr)
    std::cout << "addin given but not needed" << std::endl;

  SceneBuilder sceBui;
  sceBui.generate_scene(
      SceneBuilder::PreBuiltScene::easy_0,
      sce, cam
      );
}

void init_easy_1(
    Camera *cam, Scene  *sce,
    AdditionalInputs *addin=nullptr) {
  if (addin != nullptr)
    std::cout << "addin given but not needed" << std::endl;

  SceneBuilder sceBui;
  sceBui.generate_scene(
      SceneBuilder::PreBuiltScene::easy_1,
      sce, cam
      );
}

void init_medium_1(
    Camera *cam, Scene  *sce,
    AdditionalInputs *addin=nullptr) {
  if (addin != nullptr)
    std::cout << "addin given but not needed" << std::endl;

  SceneBuilder sceBui;
  sceBui.generate_scene(
      SceneBuilder::PreBuiltScene::medium_1,
      sce, cam
      );
}



void run_example(
	GLFWwindow* window,
  uchar4* devPtr,
  void (*init_scene)(Camera *cam, Scene  *sce, AdditionalInputs *addin),
  AdditionalInputs *addin = nullptr
  ) {

  Camera *cam = new Camera();
  Scene  *sce = new Scene();

  init_scene(cam,sce,addin);

  Renderer renderer(cam, sce);

  renderer.verbose(false);
  //renderer.benchmarking(false);


  map_resource(&devPtr);
  renderer.render(devPtr);
  unmap_resource();

  std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
  std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

	while (!glfwWindowShouldClose(window) && !NEXT) {
  //for (int i=0; i<3*30; i++)
	//while (!glfwWindowShouldClose(window))
    double elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    if (elapsed >= 1e9/30) {
      //std::cout << elapsed/1e9 << std::endl;
      map_resource(&devPtr);
      renderer.render(devPtr);
      unmap_resource();
      t0=t1;
    }
    t1 = std::chrono::steady_clock::now();

		glDrawPixels(IMG_W, IMG_H, GL_RGBA, GL_UNSIGNED_BYTE, 0);
		glfwSwapBuffers(window);
    // Poll and process events
    glfwPollEvents();
	}
  NEXT = false;
}



int main() {
  std::cout << "Some Examples!" << std::endl;

	GLFWwindow* window;
  uchar4* devPtr;

  device_setup(&window);

  AdditionalInputs *addin = new AdditionalInputs();
  for (int i=0; i<5; i++) {
    std::cout << "Example " << i << std::endl;

    addin->n_objs = addin->n_objs + 5;

    addin->n_lights = 1;
    //std::cout << "n_objs   = " << addin->n_objs << std::endl;
    //std::cout << "n_lights = " << addin->n_lights << std::endl;

    run_example(window, devPtr, init_rnd, addin);
  }

  std::cout << "Example " <<
    "easy_0" <<
    std::endl;
  run_example(window, devPtr, init_easy_0);
  std::cout << "done" << std::endl;

  std::cout << "Example " <<
    "easy_1" <<
    std::endl;
  run_example(window, devPtr, init_easy_1);
  std::cout << "done" << std::endl;

  std::cout << "Example " <<
    "medium_1" <<
    std::endl;
  run_example(window, devPtr, init_medium_1);
  std::cout << "done" << std::endl;


  // END
  device_terminate();

  return 0;
}


