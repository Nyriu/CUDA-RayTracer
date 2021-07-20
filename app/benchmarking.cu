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

  //glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE); // Hide GLFW window

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

struct AdditionalInputs {
  int n_objs   = 0;
  int n_lights = 0;
};

std::string bench_dir_path = "./times/";

BenchmarkTimeWriter* benchmark_setup() {
  if (!std::filesystem::exists(bench_dir_path)) {
    std::cout << "creating dir " << bench_dir_path << std::endl;
    std::filesystem::create_directory(bench_dir_path);
  }

  //std::string filename = "example.txt";
  int timestamp_id = std::time(0);
  std::string filename = std::to_string(timestamp_id) + ".txt";
  std::string filepath = bench_dir_path + filename;

  //std::ofstream myfile(filepath);
  //std::ofstream myfile();
  //myfile.open("example.txt");
  //std::cout << "Writing to " << filepath << std::endl;
  //myfile << "Writing this to a file.\n";
  //myfile.close();

  return new BenchmarkTimeWriter(timestamp_id, filepath);
}


void bench_init_rnd(
    Camera *cam, Scene  *sce,
    BenchmarkTimeWriter *benchfile,
    AdditionalInputs *addin=nullptr) {

  int n_objs   = 5;
  int n_lights = 1;

  if (addin != nullptr) {
    n_objs = addin->n_objs;
    n_lights = addin->n_lights;
  }

  benchfile->n_objs_   = n_objs;
  benchfile->n_lights_ = n_lights;

  SceneBuilder sceBui(n_objs, n_lights, benchfile);
  sceBui.generate_scene(
      SceneBuilder::PreBuiltScene::none,
      sce, cam
      );
}

void bench_init_empty(
    Camera *cam, Scene  *sce,
    BenchmarkTimeWriter *benchfile,
    AdditionalInputs *addin=nullptr) {
  if (addin != nullptr)
    std::cout << "bench_init_empty : addin given but not needed" << std::endl;

  SceneBuilder sceBui;
  sceBui.generate_scene(
      SceneBuilder::PreBuiltScene::none,
      sce, cam
      );
  benchfile->n_objs_   = sce->getShapesNum();
  benchfile->n_lights_ = sce->getLightsNum();
}

void bench_init_easy_0(
    Camera *cam, Scene  *sce,
    BenchmarkTimeWriter *benchfile,
    AdditionalInputs *addin=nullptr) {
  if (addin != nullptr)
    std::cout << "bench_init_easy_0 : addin given but not needed" << std::endl;

  SceneBuilder sceBui;
  sceBui.generate_scene(
      SceneBuilder::PreBuiltScene::easy_0,
      sce, cam
      );
  benchfile->n_objs_   = sce->getShapesNum();
  benchfile->n_lights_ = sce->getLightsNum();
}

void bench_init_easy_1(
    Camera *cam, Scene  *sce,
    BenchmarkTimeWriter *benchfile,
    AdditionalInputs *addin=nullptr) {
  if (addin != nullptr)
    std::cout << "bench_init_easy_1 : addin given but not needed" << std::endl;

  SceneBuilder sceBui;
  sceBui.generate_scene(
      SceneBuilder::PreBuiltScene::easy_1,
      sce, cam
      );
  benchfile->n_objs_   = sce->getShapesNum();
  benchfile->n_lights_ = sce->getLightsNum();
}

void bench_init_medium_1(
    Camera *cam, Scene  *sce,
    BenchmarkTimeWriter *benchfile,
    AdditionalInputs *addin=nullptr) {
  if (addin != nullptr)
    std::cout << "bench_init_medium_1 : addin given but not needed" << std::endl;

  SceneBuilder sceBui;
  sceBui.generate_scene(
      SceneBuilder::PreBuiltScene::medium_1,
      sce, cam
      );
  benchfile->n_objs_   = sce->getShapesNum();
  benchfile->n_lights_ = sce->getLightsNum();
}



void run_benchmarking(
	GLFWwindow* window,
  uchar4* devPtr,
  BenchmarkTimeWriter *benchfile,
  void (*bench_scene)(Camera *cam, Scene  *sce, BenchmarkTimeWriter *benchfile, AdditionalInputs *addin),
  AdditionalInputs *addin = nullptr
  ) {
  benchfile->img_h_ = IMG_H;
  benchfile->img_w_ = IMG_W;

  //std::chrono::steady_clock::time_point t0_total = std::chrono::steady_clock::now();

  Camera *cam = new Camera();
  Scene  *sce = new Scene();

  std::cout << "before bench_scene" << std::endl;
  bench_scene(cam,sce,benchfile, addin);
  std::cout << "after bench_scene" << std::endl;

  Renderer renderer(cam, sce);

  std::cout << "before map_resource" << std::endl;
  map_resource(&devPtr);
  std::cout << "after map_resource" << std::endl;
  renderer.verbose(false);
  renderer.benchmarking(true, benchfile);
  renderer.render(devPtr);
  unmap_resource();

  std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
  std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

  //int num_frames = 10;
  int num_frames = 50;
  for (int i=0; i<num_frames; i++) {
    //double elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    //if
      //(true) // uncapped "frame rate"
      // (elapsed >= 3000000)
      //(elapsed >= 16000000)
      //(elapsed >= 500000000)
      //(elapsed >= 1000000000)
      //{
      //std::cout << "\n----- elapsed = " << elapsed << "[ns]\n" << std::endl;

    t0 = std::chrono::steady_clock::now();
    map_resource(&devPtr);
    renderer.render(devPtr);
    unmap_resource();
    t1 = std::chrono::steady_clock::now();
    benchfile->total_microsec_[i] = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

      //t0=t1;
    //}
    //t1 = std::chrono::steady_clock::now();

		glDrawPixels(IMG_W, IMG_H, GL_RGBA, GL_UNSIGNED_BYTE, 0);
		glfwSwapBuffers(window);
    // Poll and process events
    glfwPollEvents();
	}
  //std::chrono::steady_clock::time_point t1_total = std::chrono::steady_clock::now();

  //benchfile->total_microsec_ = std::chrono::duration_cast<std::chrono::microseconds>(t1_total - t0_total).count();
  //benchfile->additional_ = "nada";
  benchfile->write();
}



int main() {
  std::cout << "Benchmarking!" << std::endl;
  std::chrono::steady_clock::time_point t0_benchmark_start = std::chrono::steady_clock::now();

	GLFWwindow* window;
  uchar4* devPtr;

  device_setup(&window);
  BenchmarkTimeWriter *benchfile = benchmark_setup();

  AdditionalInputs *addin = new AdditionalInputs();
  //for (int i=0; i<101; i++) {
  for (int i=0; i<51; i++) {
    std::cout << "Benchmarking " << i << std::endl;

    if (i == 0) addin->n_objs = 0;
    //else if ((i-1)% 10 == 0) addin->n_objs = addin->n_objs + 5;
    else if ((i-1)% 5 == 0) addin->n_objs = addin->n_objs + 2;
    else addin->n_objs = addin->n_objs;

    addin->n_lights = 1;
    std::cout << "n_objs   = " << addin->n_objs << std::endl;
    std::cout << "n_lights = " << addin->n_lights << std::endl;

    benchfile->id_ = i;
    run_benchmarking(window, devPtr, benchfile,
        bench_init_rnd, addin
        );
  }

  benchfile->rnd_or_enc_ = 1;
  benchfile->id_++;
  std::cout << "Benchmarking " <<
    "bench_init_empty" <<
    std::endl;
  benchfile->additional_ = "bench_init_empty";
  run_benchmarking(window, devPtr, benchfile, bench_init_empty);

  benchfile->id_++;
  std::cout << "Benchmarking " <<
    "bench_init_easy_0" <<
    std::endl;
  benchfile->additional_ = "bench_init_easy_0";
  run_benchmarking(window, devPtr, benchfile, bench_init_easy_0);
  std::cout << "done" << std::endl;

  benchfile->id_++;
  std::cout << "Benchmarking " <<
    "bench_init_easy_1" <<
    std::endl;
  benchfile->additional_ = "bench_init_easy_1";
  run_benchmarking(window, devPtr, benchfile, bench_init_easy_1);

  benchfile->id_++;
  std::cout << "Benchmarking " <<
    "bench_init_medium_1" <<
    std::endl;
  benchfile->additional_ = "bench_init_medium_1";
  run_benchmarking(window, devPtr, benchfile, bench_init_medium_1);



  // END
  benchfile->close();
  device_terminate();

  std::chrono::steady_clock::time_point t1_benchmark_end = std::chrono::steady_clock::now();
  std::cout <<
    "Benchmarking Total Time = " << std::chrono::duration_cast<std::chrono::seconds>(t1_benchmark_end - t0_benchmark_start).count() << "[s]\n" <<
    "Benchmarking Total Time = " << std::chrono::duration_cast<std::chrono::minutes>(t1_benchmark_end - t0_benchmark_start).count() << "[m]\n" << std::endl;
  return 0;
}


