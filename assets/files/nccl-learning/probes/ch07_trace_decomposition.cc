#include <cuda_runtime.h>
#include <nccl.h>
#include <nvToolsExt.h>

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    cudaError_t status_ = (call);                                                \
    if (status_ != cudaSuccess) {                                                \
      std::cerr << "CUDA failure: " << cudaGetErrorString(status_) << '\n';     \
      std::exit(10);                                                             \
    }                                                                           \
  } while (0)

#define NCCL_CHECK(call)                                                        \
  do {                                                                          \
    ncclResult_t status_ = (call);                                               \
    if (status_ != ncclSuccess) {                                                \
      std::cerr << "NCCL failure: " << ncclGetErrorString(status_) << '\n';     \
      std::exit(11);                                                             \
    }                                                                           \
  } while (0)

int main(int argc, char** argv) {
  if (argc != 5) {
    std::cerr << "usage: program direct|decomposed BYTES WARMUPS ITERATIONS\n";
    return 2;
  }
  const std::string mode = argv[1];
  const size_t bytes = std::stoull(argv[2]);
  const int warmups = std::stoi(argv[3]);
  const int iterations = std::stoi(argv[4]);
  constexpr int kGpus = 4;
  if ((mode != "direct" && mode != "decomposed") ||
      bytes % (kGpus * sizeof(int)) != 0) {
    return 3;
  }
  const size_t count = bytes / sizeof(int);
  const size_t shard_count = count / kGpus;

  ncclComm_t comms[kGpus]{};
  int devices[kGpus] = {0, 1, 2, 3};
  NCCL_CHECK(ncclCommInitAll(comms, kGpus, devices));
  cudaStream_t streams[kGpus]{};
  int* input[kGpus]{};
  int* shard[kGpus]{};
  int* output[kGpus]{};
  for (int rank = 0; rank < kGpus; ++rank) {
    CUDA_CHECK(cudaSetDevice(rank));
    CUDA_CHECK(cudaStreamCreateWithFlags(&streams[rank], cudaStreamNonBlocking));
    CUDA_CHECK(cudaMalloc(&input[rank], bytes));
    if (mode == "decomposed") {
      CUDA_CHECK(cudaMalloc(&shard[rank], bytes / kGpus));
      CUDA_CHECK(cudaMalloc(&output[rank], bytes));
    }
  }

  auto initialize = [&]() {
    for (int rank = 0; rank < kGpus; ++rank) {
      CUDA_CHECK(cudaSetDevice(rank));
      CUDA_CHECK(cudaMemsetAsync(input[rank], rank + 1, bytes, streams[rank]));
    }
  };
  auto synchronize = [&]() {
    for (int rank = 0; rank < kGpus; ++rank) {
      CUDA_CHECK(cudaSetDevice(rank));
      CUDA_CHECK(cudaStreamSynchronize(streams[rank]));
    }
  };
  auto run = [&]() {
    if (mode == "direct") {
      NCCL_CHECK(ncclGroupStart());
      for (int rank = 0; rank < kGpus; ++rank) {
        CUDA_CHECK(cudaSetDevice(rank));
        NCCL_CHECK(ncclAllReduce(input[rank], input[rank], count, ncclInt32,
                                 ncclSum, comms[rank], streams[rank]));
      }
      NCCL_CHECK(ncclGroupEnd());
    } else {
      NCCL_CHECK(ncclGroupStart());
      for (int rank = 0; rank < kGpus; ++rank) {
        CUDA_CHECK(cudaSetDevice(rank));
        NCCL_CHECK(ncclReduceScatter(input[rank], shard[rank], shard_count,
                                     ncclInt32, ncclSum, comms[rank], streams[rank]));
      }
      NCCL_CHECK(ncclGroupEnd());
      NCCL_CHECK(ncclGroupStart());
      for (int rank = 0; rank < kGpus; ++rank) {
        CUDA_CHECK(cudaSetDevice(rank));
        NCCL_CHECK(ncclAllGather(shard[rank], output[rank], shard_count,
                                 ncclInt32, comms[rank], streams[rank]));
      }
      NCCL_CHECK(ncclGroupEnd());
    }
  };

  for (int index = 0; index < warmups; ++index) {
    initialize();
    run();
    synchronize();
  }

  nvtxRangePushA(mode.c_str());
  for (int index = 0; index < iterations; ++index) {
    initialize();
    run();
    synchronize();
  }
  nvtxRangePop();

  const int expected = 0x0a0a0a0a;
  bool correct = true;
  for (int rank = 0; rank < kGpus; ++rank) {
    CUDA_CHECK(cudaSetDevice(rank));
    int first = 0;
    int last = 0;
    int* result = mode == "direct" ? input[rank] : output[rank];
    CUDA_CHECK(cudaMemcpy(&first, result, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&last, result + count - 1, sizeof(int), cudaMemcpyDeviceToHost));
    correct = correct && first == expected && last == expected;
  }
  std::cout << "mode=" << mode << " bytes=" << bytes
            << " iterations=" << iterations << " correct=" << correct << '\n';

  for (int rank = 0; rank < kGpus; ++rank) {
    CUDA_CHECK(cudaSetDevice(rank));
    if (output[rank]) CUDA_CHECK(cudaFree(output[rank]));
    if (shard[rank]) CUDA_CHECK(cudaFree(shard[rank]));
    CUDA_CHECK(cudaFree(input[rank]));
    CUDA_CHECK(cudaStreamDestroy(streams[rank]));
    NCCL_CHECK(ncclCommDestroy(comms[rank]));
  }
  return correct ? 0 : 4;
}
