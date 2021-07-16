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

  glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);

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


/*
   Valori interessanti da misurare
   * Tempo totale da lancio a conclusione esempio
   * Tempo generazione singoli frame
   * Tempo update scena
   * Utilizzo memoria

   * nvprof come dove quando?
   * usare eventi cuda?

   Al variare di
   * numero oggetti
   * numero luci
   * dimensione finestra
   * update abilitato/disabilitato (1 solo frame)
*/

void empty_scene(
    ) {
}



int main() {
  std::cout << "Benchmarking!" << std::endl;

	GLFWwindow* window;
  uchar4* devPtr;

  device_setup(&window);

  Camera *cam = new Camera();
  Scene  *sce = new Scene();

  int n_objs   = 5; // objects number in random scene
  int n_lights = 1; // objects number in random scene
  SceneBuilder sceBui(n_objs, n_lights);
  sceBui.generate_scene(
      SceneBuilder::PreBuiltScene::none,
      sce, cam
      );

  Renderer renderer(cam, sce);

  map_resource(&devPtr);
  renderer.verbose(false);
  renderer.render(devPtr);
  unmap_resource();

  std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
  std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();


  int num_frames = 10;
  for (int i=0; i<num_frames; i++) {
    double elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    if
      (true) // uncapped "frame rate"
      // (elapsed >= 3000000)
      //(elapsed >= 16000000)
      //(elapsed >= 500000000)
      //(elapsed >= 1000000000)
      {
      //std::cout << "\n----- elapsed = " << elapsed << "[ns]\n" << std::endl;

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

  device_terminate();

  return 0;
}


