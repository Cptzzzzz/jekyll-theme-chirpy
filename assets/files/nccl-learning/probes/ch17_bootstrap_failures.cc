#include <cuda_runtime.h>
#include <nccl.h>

#include <cstring>
#include <iostream>

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: ch17_bootstrap_failures MODE\n";
    return 2;
  }

  if (std::strcmp(argv[1], "missing-rank") == 0) {
    ncclUniqueId id{};
    ncclResult_t result = ncclGetUniqueId(&id);
    if (result != ncclSuccess) return 3;
    if (cudaSetDevice(0) != cudaSuccess) return 4;
    std::cout << "MISSING_RANK entering ncclCommInitRank rank=0 nranks=2"
              << std::endl;
    ncclComm_t comm = nullptr;
    result = ncclCommInitRank(&comm, 2, id, 0);
    std::cout << "MISSING_RANK unexpected return result="
              << ncclGetErrorString(result) << std::endl;
    if (comm) ncclCommAbort(comm);
    return result == ncclSuccess ? 5 : 6;
  }

  if (std::strcmp(argv[1], "duplicate-gpu") == 0) {
    int devices[2] = {0, 0};
    ncclComm_t comms[2] = {nullptr, nullptr};
    std::cout << "DUPLICATE_GPU entering ncclCommInitAll devices=0,0"
              << std::endl;
    ncclResult_t result = ncclCommInitAll(comms, 2, devices);
    std::cout << "DUPLICATE_GPU result=" << ncclGetErrorString(result)
              << " code=" << static_cast<int>(result) << std::endl;
    for (ncclComm_t comm : comms) {
      if (comm) ncclCommAbort(comm);
    }
    return result == ncclInvalidUsage ? 0 : 7;
  }

  std::cerr << "unknown mode: " << argv[1] << '\n';
  return 2;
}
