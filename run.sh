#!/bin/sh

if [ "$1" == "-bb" ]; then
  rm -rf ./build ./executable
fi

if [[ -d "./build" || -d "./executable" ]]; then
  rm -rf executable/*
  cd build
  make -j
  cd ..
elif [ ! -f "./executable/main" ]; then
  echo "Building..."
  rm -rf build executable/*
  mkdir build
  cd build
  cmake -DCMAKE_CUDA_COMPILER=/opt/cuda/bin/nvcc ..
  #cmake -DCMAKE_CUDA_COMPILER=/opt/cuda/bin/nvcc -DCMAKE_BUILD_TYPE=Debug ..
  #cmake -DCMAKE_CUDA_COMPILER=/opt/cuda/bin/nvcc -DCMAKE_BUILD_TYPE=Release ..
  #-DCMAKE_BUILD_TYPE=Debug | Coverage | Release]
  make -j
  cd ..
fi


#if [ "$1" == "-t" ]; then
#  echo "With 5 sec timeout"
#  timeout 5 ./executable/main

if [ "$1" == "-b" ]; then
  ./executable/benchmarking
elif [ "$1" == "-e" ]; then
  ./executable/example
else
  ./executable/main
fi

