#ifndef COMMON_H
#define COMMON_H

#include <iostream>
#include <fstream>
#include <filesystem>

#include <glm/glm.hpp>
//#include <glm/common.hpp>
#include <glm/vec3.hpp>
#include <string>
//#include <glm/fwd.hpp>
//#include <glm/geometric.hpp>


// Globals
#define IMG_H 512
#define IMG_W 512

// TODO
//#define IMG_H 1080
//#define IMG_W IMG_H*16/9;


static void HandleError( cudaError_t err, const char *file, int line) {
  if (err != cudaSuccess) {
    std::cout << "Error Name: " << cudaGetErrorName( err ) << std::endl;
    std::cout << cudaGetErrorString( err ) << " in " << file << " line " << line << std::endl;
    exit(EXIT_FAILURE);
  }
}
#define HANDLE_ERROR(err)(HandleError(err, __FILE__, __LINE__))


constexpr float infinity = std::numeric_limits<float>::max();


// Aliases
using vec3   = glm::vec3;
using point3 = glm::vec3;
using color  = glm::vec3;


// Benchmarking stuff
class BenchmarkTimeWriter {
  private:
    int timestamp_id_ = 0;
    std::string filepath_;
    std::ofstream myfile_;

  public:
    int img_h_;
    int img_w_;
    int n_objs_;
    int n_lights_;
    int rnd_or_enc_;
    int seed_or_code_;
    int total_microsec_;
    int update_[50] = { 0 };
    int render_[50] = { 0 };
    std::string additional_;
  public:
    BenchmarkTimeWriter() = default;
    BenchmarkTimeWriter(int timestamp_id, std::string filepath) :
      timestamp_id_(timestamp_id), filepath_(filepath) {
        std::cout << filepath_ << std::endl;
        myfile_.open(filepath_);
        myfile_ << "IMG_H" << ",";
        myfile_ << "IMG_W" << ",";
        myfile_ << "n_obj" << ",";
        myfile_ << "n_lights" << ",";
        myfile_ << "rnd-enc" << ",";
        myfile_ << "seed-code" << ",";
        myfile_ << "total_µs" << ",";
        for (int i=1; i<=50; i++) {
          myfile_ << "update_" << i << ",";
          myfile_ << "render_" << i << ",";
        }
        myfile_ << "additional" << "\n";
      }


    void write() {
      myfile_ << img_h_     << ",";
      myfile_ << img_w_     << ",";
      myfile_ << n_objs_     << ",";
      myfile_ << n_lights_  << ",";
      myfile_ << rnd_or_enc_ << ",";
      myfile_ << seed_or_code_ << ",";
      myfile_ << total_microsec_  << ",";
      for (int i=1; i<=50; i++) {
        myfile_ << update_[i] << ",";
        myfile_ << render_[i] << ",";
      }
      myfile_ << additional_ << "\n";
    }

    void close() { myfile_.close(); }
};


#endif
