#include "Scene.h"
#include <iostream>
#include <ostream>
#include <cooperative_groups.h>

void Scene::addShape(ImplicitShape* shape) {
  shapes_num_++;
  shapes_.push_back(shape);
}


void Scene::addLight(Light* light) {
  lights_num_++;
  lights_.push_back(light);
}

void Scene::addAmbientLight(AmbientLight* light) {
  ambientLight_ = light;
}

__device__ ImplicitShape *d_shapes = nullptr;
__device__ int d_n_shapes = 0;
static __global__ void copy_kernel(ImplicitShapeInfo *infos, int n_shapes) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  if (x >= n_shapes) return;
  if (x == 0) {
    // only the first thread inits vars
    //printf("(%d) init...\n", x);
    size_t pols_size = sizeof(ImplicitShape)*n_shapes;
    d_shapes = (ImplicitShape*) malloc(pols_size);
    d_n_shapes = n_shapes;
    //printf("(%d) d_shapes=%p\n", x,d_shapes);
  }
  //__syncthreads(); // all threads must wait
  cooperative_groups::this_thread_block().sync(); // alternative // does the same

  ImplicitShape *sh_ptr = nullptr;
  ImplicitShapeInfo *isi_ptr = infos+x;

  //printf("(%d) type=%d\n", x, isi_ptr->shape_type);

  if (isi_ptr->shape_type == ShapeType::sphere) {
    //printf("(%d) we've got a sphere!\n", x);
    sh_ptr = new Sphere(*isi_ptr);
  } else if (isi_ptr->shape_type == ShapeType::none) {
    //printf("(%d) we've got a none!\n", x);
    sh_ptr = new ImplicitShape(*isi_ptr);
  } else {
    printf("(%d) we've got a PROBLEM!\n", x);
  }
  memcpy(d_shapes+x, sh_ptr, sizeof(*sh_ptr));
}

__device__ ImplicitShape* Scene::getShapes() const { return d_shapes; }




__host__ void Scene::shapes_to_device() {
  size_t infos_size = sizeof(ImplicitShapeInfo)*shapes_.size();
  ImplicitShapeInfo *infos = (ImplicitShapeInfo *) malloc(infos_size);
  int i = 0;
  for (const ImplicitShape *sh : shapes_) {
    ImplicitShapeInfo isi = sh->get_info();
    memcpy(&infos[i], &isi, sizeof(isi));
    i++;
  }

  ImplicitShapeInfo *dev_infos = nullptr;
  HANDLE_ERROR(
      cudaMalloc((void**)&dev_infos, infos_size)
      );
  HANDLE_ERROR(
      cudaMemcpy((void*)dev_infos, (void*)infos, infos_size, cudaMemcpyHostToDevice)
      );

  free(infos);
  copy_kernel<<<1,shapes_.size()>>>(dev_infos, shapes_.size());
  HANDLE_ERROR(cudaDeviceSynchronize());
  //std::cout << "here" <<  std::endl;
  //std::cout << devShapes_ <<  std::endl;
  //std::cout << d_shapes <<  std::endl;
  //std::cout << sizeof(d_shapes) <<  std::endl;
  //HANDLE_ERROR(
  //    cudaMemcpy((void*)devShapes_, (void*)d_shapes, sizeof(&devShapes_), cudaMemcpyDeviceToHost)
  //    );
  //std::cout << "after here" <<  std::endl;

  /// OLD STUFF BELOW
  /// PROBABLY REMOVE ALL
  /// /// DEBUG
  /// std::cout << 
  ///   "\nsizeof(SceneObject) = " << sizeof(SceneObject) << 
  ///   "\nsizeof(ImplicitShape) = " << sizeof(ImplicitShape) << 
  ///   "\nsizeof(Sphere) = " << sizeof(Sphere) << 
  ///   std::endl;
  /// /// END /// DEBUG


  /// size_t total_size = 0;
  /// for (const ImplicitShape *shape : shapes_) {
  ///   total_size += sizeof(*shape);
  ///   std::cout << 
  ///     "\nsizeof(shape) = " << sizeof(*shape) << 
  ///     std::endl;
  /// }
  /// // Static allocation on device memory
  /// HANDLE_ERROR(
  ///     cudaMalloc((void**)&devShapes_, total_size)
  ///     );

  /// int offset = 0;
  /// for (const ImplicitShape *shape : shapes_) {
  ///   // Copy from host to device
  ///   HANDLE_ERROR(
  ///       cudaMemcpy((void*)(devShapes_+offset), (void*)shape, sizeof(*shape), cudaMemcpyHostToDevice)
  ///       );
  ///   offset++;
  /// }

  /// if (offset != shapes_num_) {
  ///   std::cout << "ERROR"
  ///     "offset = " << offset <<
  ///     "shapes_num_ = " << shapes_num_ <<
  ///     std::endl;
  ///   exit(1);
  /// }
}

__host__ void Scene::lights_to_device() {
  if (lights_num_ > 0) {
    size_t total_size = 0;
    for (Light *lgt : lights_) {
      total_size += sizeof(*lgt);
    }
    // Static allocation on device memory
    HANDLE_ERROR(
        cudaMalloc((void**)&devLights_, total_size)
        );

    int offset = 0;
    for (Light *lgt : lights_) {
      // Copy from host to device
      HANDLE_ERROR(
          cudaMemcpy((void*)(devLights_+offset), (void*)lgt, sizeof(*lgt), cudaMemcpyHostToDevice)
          );
      offset++;
    }

    if (offset != lights_num_) {
      std::cout << "ERROR"
        "offset = " << offset <<
        "lights_num_ = " << lights_num_ <<
        std::endl;
      exit(1);
    }
  }
  if (hasAmbientLight()) {
    HANDLE_ERROR(
        cudaMalloc((void**)&devAmbLight_, sizeof(AmbientLight))
        );
    HANDLE_ERROR(
        cudaMemcpy((void*)devAmbLight_, (void*)ambientLight_, sizeof(AmbientLight), cudaMemcpyHostToDevice)
        );
  }
}

__host__ Scene* Scene::to_device() {
  if (shapes_num_ > 0) {
    shapes_to_device();
  }
  if (lights_num_ > 0) {
    lights_to_device();
  }

  // Static allocation on device memory
  HANDLE_ERROR(
      cudaMalloc((void**)&devPtr_, sizeof(Scene))
      );
  // Copy from host to device
  HANDLE_ERROR(
      cudaMemcpy((void*)devPtr_, (void*)this, sizeof(Scene), cudaMemcpyHostToDevice)
      );
  return devPtr_;
}
