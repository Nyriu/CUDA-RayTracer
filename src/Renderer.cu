#include "Renderer.h"
#include "common.h"
#include <chrono>
#include <iostream>


static __global__ void render_kernel(uchar4 *ptr,
    const Camera *cam,
    const Scene *sce,
    const Tracer *trc//, const float2 *AA_array, const int AA_array_len
    ) {

  // map from threadIdx/BlockIdx to pixel positioa
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  //int offset = x + y * blockDim.x * gridDim.x;
  int offset = x + y * IMG_W;

  if (offset >= IMG_H*IMG_H) return;

  // in img coord (0,0) is bottom-left
  // Put coords in [0,1]
  float u = (x + .5) / ((float) IMG_W -1); // NDC Coord
  float v = (y + .5) / ((float) IMG_H -1); // NDC Coord

  // base color
  Ray r = cam->generate_ray(u,v);
  color c = trc->trace(&r, sce);

  /// // TENTATIVO AA
  ///for (int i=0; i<AA_array_len; i++) {il passaggio del buffer ad OpenGL.
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

static __global__ void kernel_update_scene(Scene *sce) {
  if (
      threadIdx.x + blockIdx.x * blockDim.x +
      threadIdx.y + blockIdx.y * blockDim.y +
      threadIdx.z + blockIdx.z * blockDim.z == 0) {
    sce->update();
  }
}

__host__ Renderer::Renderer(
    Camera *cam,
    Scene *sce,
    int max_num_tick
    ) :
  max_num_tick_(max_num_tick) {
  if (
      devCamPtr_ == nullptr ||
      devScePtr_ == nullptr //|| devTrcPtr_ == nullptr)
    ) { // a bit ugly...
    HANDLE_ERROR(
        cudaMalloc((void**)&devCamPtr_, sizeof(Camera))
        );
    HANDLE_ERROR(
        cudaMemcpy((void*)devCamPtr_, (void*)cam, sizeof(Camera), cudaMemcpyHostToDevice)
        );
    devScePtr_ = sce->to_device();
  }
}

__host__ void Renderer::render(uchar4 *devPtr) {
  // --- Generate One Frame ---

  //dim3 grids(IMG_W, IMG_H);
  //dim3 threads(1);

  dim3 grids(IMG_W/16, IMG_H/16);
  dim3 threads(16,16);

  //dim3 grids(32,32); // same as the one above
  //dim3 threads(16,16);

  //dim3 grids(30,30);
  //dim3 threads(18,18);

  //cudaDeviceProp prop;
  //int count;
  //HANDLE_ERROR( cudaGetDeviceCount( &count) );
  //if (count < 0) exit(0); // TODO handle better
  //HANDLE_ERROR( cudaGetDeviceProperties( &prop, 0 ));
  //int sm_count = prop.multiProcessorCount;
  //dim3 grids(sm_count);
  //dim3 threads(IMG_W, IMG_H);

  if (benchmarking_) {
    bTWriter_->grids_ = grids;
    bTWriter_->threads_ = threads;
  }

  if (!done_cuda_free_ && current_tick_ == max_num_tick_) {
    HANDLE_ERROR(cudaFree((void*)devCamPtr_));
    HANDLE_ERROR(cudaFree((void*)devScePtr_));
    //HANDLE_ERROR(cudaFree((void*)devTrcPtr_)); // TODO
    done_cuda_free_ = true;
    return;
  }
  if (done_cuda_free_) return;

  if (
      devCamPtr_ == nullptr ||
      devScePtr_ == nullptr //|| devTrcPtr_ == nullptr
     ) { // a bit ugly
    std::cout << "\nRenderer::render : ERROR bad device initialization?" << std::endl;
  }

  cudaEvent_t start_frame, stop_frame;
  HANDLE_ERROR(cudaEventCreate(&start_frame));
  HANDLE_ERROR(cudaEventCreate(&stop_frame));


  cudaEvent_t start_update, stop_update;
  HANDLE_ERROR(cudaEventCreate(&start_update));
  HANDLE_ERROR(cudaEventCreate(&stop_update));

  std::chrono::steady_clock::time_point t0_update = std::chrono::steady_clock::now();
  HANDLE_ERROR(cudaEventRecord(start_frame));
  HANDLE_ERROR(cudaEventRecord(start_update));

  kernel_update_scene<<<1,1>>>(devScePtr_);

  HANDLE_ERROR(cudaEventRecord(stop_update));
  HANDLE_ERROR(cudaEventSynchronize(stop_update));
  //HANDLE_ERROR(cudaDeviceSynchronize());
  // qua t1
  std::chrono::steady_clock::time_point t1_update = std::chrono::steady_clock::now();

  float update_time = .0f;
  HANDLE_ERROR(cudaEventElapsedTime(&update_time, start_update, stop_update));
  update_time *= 1000; // ms to microseconds

  if (verbose_) {
    std::cout <<
      "Update Time = " << std::chrono::duration_cast<std::chrono::     seconds>(t1_update - t0_update).count() << "[s]\n" <<
      "Update Time = " << std::chrono::duration_cast<std::chrono::microseconds>(t1_update - t0_update).count() << "[µs]\n" <<
      "(CUDAEvent) Update Time = " << update_time << "[µs]\n" <<
      std::endl;
  }
  if (benchmarking_) {
    bTWriter_->update_[current_tick_] = std::chrono::duration_cast<std::chrono::microseconds>(t1_update - t0_update).count();
    bTWriter_->cudae_update_[current_tick_] = update_time;
  }


  {
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
  }

  // qua t0
  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  //std::cout <<
  //  "generating frame num " <<
  //  current_tick_ << "\n" << std::endl;

  //cudaStream_t stream;
  //cudaStreamCreate(&stream);
  cudaEvent_t start_render, stop_render;
  HANDLE_ERROR(cudaEventCreate(&start_render));
  HANDLE_ERROR(cudaEventCreate(&stop_render));


  //cudaEventRecord(start, stream);
  HANDLE_ERROR(cudaEventRecord(start_render));
  render_kernel<<<grids,threads>>>(devPtr, devCamPtr_, devScePtr_, devTrcPtr_//,
  //render_kernel<<<grids,threads,0,stream>>>(devPtr, devCamPtr_, devScePtr_, devTrcPtr_//,
      //dev_AA_array, AA_array_len
      );
  HANDLE_ERROR(cudaEventRecord(stop_render));
  HANDLE_ERROR(cudaEventRecord(stop_frame));
  HANDLE_ERROR(cudaEventSynchronize(stop_render));
  HANDLE_ERROR(cudaEventSynchronize(stop_frame));
  //HANDLE_ERROR(cudaDeviceSynchronize());
  // qua t1
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

  float render_time = .0f;
  HANDLE_ERROR(cudaEventElapsedTime(&render_time, start_render, stop_render));
  render_time *= 1000; // ms to microseconds

  float frame_time = .0f; // update+render
  HANDLE_ERROR(cudaEventElapsedTime(&frame_time, start_frame, stop_frame));
  frame_time *= 1000; // ms to microseconds

  if (verbose_) {
    std::cout <<
      "Frame Num = " << current_tick_ << "\n" << 
      "Render Time = " << std::chrono::duration_cast<std::chrono::     seconds>(end - begin).count() << "[s]\n" <<
      "Render Time = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]\n" << //std::endl;
      "(CUDAEvent) Render Time = " << render_time << "[µs]\n" <<
      "\n(CUDAEvent) Frame Time = " << frame_time << "[µs]\n" << std::endl;
  }
  if (benchmarking_) {
    bTWriter_->render_[current_tick_] = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    bTWriter_->cudae_render_[current_tick_] = render_time;
    bTWriter_->cudae_frame_[current_tick_] = frame_time;
  }


  //HANDLE_ERROR(cudaFree((void*)devCamPtr_));
  //HANDLE_ERROR(cudaFree((void*)devScePtr_));
  //HANDLE_ERROR(cudaFree((void*)devTrcPtr_));

  //HANDLE_ERROR(cudaDeviceSynchronize());

  current_tick_++;
}

