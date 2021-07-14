#include "Renderer.h"
#include "common.h"
#include <chrono>


static __global__ void kernel(uchar4 *ptr,
    const Camera *cam,
    const Scene *sce,
    const Tracer *trc//, const float2 *AA_array, const int AA_array_len
    ) {

  // map from threadIdx/BlockIdx to pixel position
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int offset = x + y * blockDim.x * gridDim.x;

  // in img coord (0,0) is bottom-left
  // Put coords in [0,1]
  float u = (x + .5) / ((float) IMG_W -1); // NDC Coord
  float v = (y + .5) / ((float) IMG_H -1); // NDC Coord

  // base color
  Ray r = cam->generate_ray(u,v);
  color c = trc->trace(&r, sce);

  /// // TENTATIVO AA
  ///for (int i=0; i<AA_array_len; i++) {
  ///  Ray aar = cam->generate_ray(
  ///      (x + .5 + AA_array[i].x) / ((float) IMG_W -1),
  ///      (y + .5 + AA_array[i].y) / ((float) IMG_H -1)
  ///      );
  ///  c += trc->trace(&aar, sce);
  ///}
  ///c /= color(AA_array_len+1);
  /// END // TENTATIVO AA

  //color c(0.2);


  // accessing uchar4 vs unsigned char*
  ptr[offset].x = (int) (255 * c.r); // (int) (u * 255); //0;
  ptr[offset].y = (int) (255 * c.g); // (int) (v * 255); //(int)255/2;
  ptr[offset].z = (int) (255 * c.b); // 0;
  ptr[offset].w = 255;
}


__host__ void Renderer::render(
    Camera *cam,
    Scene *sce,
    uchar4 *devPtr) {
  // --- Generate One Frame ---
  // TODO dims
  dim3 grids(IMG_W/16, IMG_H/16);
  dim3 threads(16,16);
  //dim3 grids(IMG_W, IMG_H);
  //dim3 threads(1);
  //float grids = 1;
  //dim3 threads(IMG_W, IMG_H);

  Camera *devCamPtr = nullptr;
  Tracer *devTrcPtr = nullptr; // TODO

  // Static allocation on device memory
  HANDLE_ERROR(
      cudaMalloc((void**)&devCamPtr, sizeof(Camera))
      );
  // Copy from host to device
  HANDLE_ERROR(
      cudaMemcpy((void*)devCamPtr, (void*)cam, sizeof(Camera), cudaMemcpyHostToDevice)
      );
  Scene *devScePtr = sce->to_device();

  /// TENTATIVO AA
  /// int AA_array_len = 8;
  /// size_t AA_array_size = sizeof(float2) * AA_array_len;
  /// float2 *AA_array = (float2 *) malloc(AA_array_size);
  /// float val = 0; //0.0000001;
  /// AA_array[0] = make_float2(-val, -val);
  /// AA_array[1] = make_float2(-val,  val);
  /// AA_array[2] = make_float2( val, -val);
  /// AA_array[3] = make_float2( val,  val);

  /// AA_array[0+4] = make_float2(AA_array[0+4].x/2.f, AA_array[0+4].y/2.f);
  /// AA_array[1+4] = make_float2(AA_array[1+4].x/2.f, AA_array[1+4].y/2.f);
  /// AA_array[2+4] = make_float2(AA_array[2+4].x/2.f, AA_array[2+4].y/2.f);
  /// AA_array[3+4] = make_float2(AA_array[3+4].x/2.f, AA_array[3+4].y/2.f);

  /// float2 *dev_AA_array = nullptr;
  /// HANDLE_ERROR( cudaMalloc((void**)&dev_AA_array, AA_array_size) );
  /// HANDLE_ERROR(
  ///     cudaMemcpy((void*)dev_AA_array, (void*)AA_array, AA_array_size, cudaMemcpyHostToDevice)
  ///     );
  /// END // TENTATIVO AA



  // qua t0
  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  //std::cout <<
  //  "generating frame num " <<
  //  current_tick_ << "\n" << std::endl;

  kernel<<<grids,threads>>>(devPtr, devCamPtr, devScePtr, devTrcPtr//,
      //dev_AA_array, AA_array_len
      );
  HANDLE_ERROR(cudaDeviceSynchronize());
  // qua t1
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  // per calcolare quanto ha messo a fare il frame
  std::cout << "Frame Gen Time = " << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count() << "[s]" << std::endl;
  std::cout << "Frame Gen Time = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;


  HANDLE_ERROR(cudaFree((void*)devCamPtr));
  HANDLE_ERROR(cudaFree((void*)devScePtr));
  HANDLE_ERROR(cudaFree((void*)devTrcPtr));

  HANDLE_ERROR(cudaDeviceSynchronize());
}

