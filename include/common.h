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
    int id_=0;
    int img_h_;
    int img_w_;
    int n_objs_;
    int n_lights_;
    int rnd_or_enc_=0;
    int seed_or_code_;
    //int total_microsec_;
    int total_microsec_[50] = { 0 };
    int update_[50] = { 0 }; // host timing
    int render_[50] = { 0 }; // host timing
    int cudae_update_[50] = { 0 }; // cuda event timing
    int cudae_render_[50] = { 0 }; // cuda event timing
    int cudae_frame_[50] = { 0 }; // cuda event timing
    std::string additional_;
  public:
    BenchmarkTimeWriter() = default;
    BenchmarkTimeWriter(int timestamp_id, std::string filepath) :
      timestamp_id_(timestamp_id), filepath_(filepath) {
        std::cout << filepath_ << std::endl;
        myfile_.open(filepath_);
        myfile_ << "id" << ",";
        myfile_ << "img_h" << ",";
        myfile_ << "img_w" << ",";
        myfile_ << "n_objs" << ",";
        myfile_ << "n_lights" << ",";
        myfile_ << "rnd_or_enc" << ",";
        myfile_ << "seed_or_code" << ",";
        myfile_ << "frame_num" << ",";
        myfile_ << "total_microsec" << ",";
        myfile_ << "update_time" << ",";
        myfile_ << "render_time" << ",";
        //for (int i=1; i<=50; i++) {
        //  myfile_ << "update_" << i << ",";
        //  myfile_ << "render_" << i << ",";
        //}
        myfile_ << "cudae_frame" << ",";
        myfile_ << "cudae_update" << ",";
        myfile_ << "cudae_render" << ",";

        myfile_ << "additional" << "\n";
      }


    void write() {
      for (int i=0; i<50; i++) {
        myfile_ << id_     << ",";
        myfile_ << img_h_     << ",";
        myfile_ << img_w_     << ",";
        myfile_ << n_objs_     << ",";
        myfile_ << n_lights_  << ",";
        myfile_ << rnd_or_enc_ << ",";
        myfile_ << seed_or_code_ << ",";
        myfile_ << i+1  << ",";
        //myfile_ << total_microsec_  << ",";
        myfile_ << total_microsec_[i] << ",";
        myfile_ << update_[i] << ",";
        myfile_ << render_[i] << ",";

        myfile_ << cudae_update_[i] << ",";
        myfile_ << cudae_render_[i] << ",";
        myfile_ << cudae_frame_ [i] << ",";

        myfile_ << additional_ << "\n";
      }
    }

    void close() { myfile_.close(); }
};


#endif
