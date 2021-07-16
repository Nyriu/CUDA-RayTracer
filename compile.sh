#!/bin/sh
rm -rf build executable
mkdir build
cd build
cmake -D CMAKE_CUDA_COMPILER=/opt/cuda/bin/nvcc ..
if [ "$1" == "-j" ]; then
  make -j
else
  make
fi
cd ..
