#ifndef COMMON_H
#define COMMON_H

#include <iostream>

#include <glm/glm.hpp>
//#include <glm/common.hpp>
#include <glm/vec3.hpp>
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

#endif
