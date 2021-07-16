#include <glm/geometric.hpp>
#define GL_GLEXT_PROTOTYPES

#include <iostream>
#include <chrono>

#include <GLFW/glfw3.h>

#include <cuda.h>
#include <cuda_gl_interop.h>

#include "common.h"
#include "Renderer.h"
#include "SceneBuilder.h"


static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
  // Press ESC to close the window
  if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
    glfwSetWindowShouldClose(window, GLFW_TRUE);
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


int main() {
	GLFWwindow* window;
  uchar4* devPtr;

  device_setup(&window);

  Camera cam;
  //Camera cam(point3(0,0,5));
  Scene sce;

  SceneBuilder sceBui(5,1);
  sceBui.generate_scene(
      &sce, &cam,
      //SceneBuilder::PreBuiltScene::none
      SceneBuilder::PreBuiltScene::simple_moving
      );

  Renderer renderer(&cam, &sce);
  //Renderer renderer(&cam, &sce, 3); // limit num of generated frames

  map_resource(&devPtr);
  renderer.render(devPtr);
  unmap_resource();

  std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
  std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();


	while (!glfwWindowShouldClose(window)) {
    double elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    //if (elapsed >= 3000000) {
    if (elapsed >= 16000000) {
    //if (elapsed >= 500000000) {
    //if (elapsed >= 1000000000) {
      std::cout << "\n----- elapsed = " << elapsed << "[ns]\n" << std::endl;

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
    // Here update scene to be interactive // TODO
	}

  device_terminate();

  return 0;
}

