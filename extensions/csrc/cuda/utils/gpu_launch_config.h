#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

namespace colossalAI {
namespace cuda {
namespace utils {

class NVGPUDevInfo;
class GPULaunchConfig;

GPULaunchConfig GetGPULaunchConfig1D(const NVGPUDevInfo& dev_info,
                                     int64_t numel, int64_t vec_size);

// TODO(LiuYang): to be implemented
GPULaunchConfig GetGPULaunchConfig2D(const NVGPUDevInfo& dev_info,
                                     int64_t numel, int64_t vec_size);

// TODO(LiuYang): to be implemented
GPULaunchConfig GetGPULaunchConfig3D(const NVGPUDevInfo& dev_info,
                                     int64_t numel, int64_t vec_size);

class GPULaunchConfig {
 public:
  GPULaunchConfig(){};
  GPULaunchConfig(const dim3& block, const dim3& grid)
      : block_(block), grid_(grid) {}
  dim3 get_block() const { return block_; }
  dim3 get_grid() const { return grid_; }
  friend GPULaunchConfig GetGPULaunchConfig1D(const NVGPUDevInfo& dev_info,
                                              int64_t numel, int64_t vec_size);

 protected:
  void set_block(const dim3& dim) { block_ = dim; }
  void set_grid(const dim3& dim) { grid_ = dim; }

 private:
  dim3 block_{1, 1, 1};
  dim3 grid_{1, 1, 1};
};

}  // namespace utils
}  // namespace cuda
}  // namespace colossalAI
