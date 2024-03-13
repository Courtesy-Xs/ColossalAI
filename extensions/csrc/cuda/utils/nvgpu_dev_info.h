#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <array>

#include "micros.h"

namespace colossalAI {
namespace cuda {
namespace utils {

class NVGPUDevInfo {
 public:
  explicit NVGPUDevInfo(int device_num) : device_num_(device_num) {
    CUDA_CHECK(cudaGetDeviceProperties(prop_, device_num));
  }

  std::array<int, 3> GetMaxGridDims() const;
  std::array<int, 3> GetMaxBlockDims() const;
  std::array<int, 2> GetCapability() const;
  int GetMultiProcessorCount() const;
  int GetMaxThreadsPerMultiProcessor() const;
  int GetMaxThreadsPerBlock() const;

 private:
  int device_num_;
  cudaDeviceProp* prop_;
};

}  // namespace utils
}  // namespace cuda
}  // namespace colossalAI
