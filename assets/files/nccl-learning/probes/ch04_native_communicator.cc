#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <unistd.h>

#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    cudaError_t status_ = (call);                                                \
    if (status_ != cudaSuccess) {                                                \
      std::cerr << "CUDA failure " << __FILE__ << ":" << __LINE__ << ": "      \
                << cudaGetErrorString(status_) << std::endl;                    \
      MPI_Abort(MPI_COMM_WORLD, 10);                                             \
    }                                                                           \
  } while (0)

#define NCCL_CHECK(call)                                                        \
  do {                                                                          \
    ncclResult_t status_ = (call);                                               \
    if (status_ != ncclSuccess) {                                                \
      std::cerr << "NCCL failure " << __FILE__ << ":" << __LINE__ << ": "      \
                << ncclGetErrorString(status_) << std::endl;                    \
      MPI_Abort(MPI_COMM_WORLD, 11);                                             \
    }                                                                           \
  } while (0)

static uint64_t hash_unique_id(const ncclUniqueId& id) {
  const auto* bytes = reinterpret_cast<const unsigned char*>(&id);
  uint64_t hash = 1469598103934665603ULL;
  for (size_t index = 0; index < sizeof(id); ++index) {
    hash ^= bytes[index];
    hash *= 1099511628211ULL;
  }
  return hash;
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int global_rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  if (world_size != 4 || argc != 2) {
    if (global_rank == 0) std::cerr << "usage: mpirun -np 4 program OUTPUT_DIR" << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 2);
  }

  int device_count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&device_count));
  if (device_count < world_size) MPI_Abort(MPI_COMM_WORLD, 3);
  const int cuda_device = global_rank;
  CUDA_CHECK(cudaSetDevice(cuda_device));

  char pci_bus_id[32] = {0};
  CUDA_CHECK(cudaDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), cuda_device));

  ncclUniqueId unique_id{};
  if (global_rank == 0) NCCL_CHECK(ncclGetUniqueId(&unique_id));
  MPI_Bcast(&unique_id, sizeof(unique_id), MPI_BYTE, 0, MPI_COMM_WORLD);
  const uint64_t unique_id_hash = hash_unique_id(unique_id);

  MPI_Barrier(MPI_COMM_WORLD);
  const double init_start = MPI_Wtime();
  ncclComm_t world_comm = nullptr;
  NCCL_CHECK(ncclCommInitRank(&world_comm, world_size, unique_id, global_rank));
  const double init_ms = (MPI_Wtime() - init_start) * 1000.0;

  int comm_count = -1;
  int comm_rank = -1;
  int comm_device = -1;
  NCCL_CHECK(ncclCommCount(world_comm, &comm_count));
  NCCL_CHECK(ncclCommUserRank(world_comm, &comm_rank));
  NCCL_CHECK(ncclCommCuDevice(world_comm, &comm_device));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  float* device_value = nullptr;
  CUDA_CHECK(cudaMalloc(&device_value, sizeof(float)));
  float world_value = static_cast<float>(global_rank + 1);
  CUDA_CHECK(cudaMemcpyAsync(device_value, &world_value, sizeof(float), cudaMemcpyHostToDevice, stream));
  NCCL_CHECK(ncclAllReduce(device_value, device_value, 1, ncclFloat, ncclSum, world_comm, stream));
  CUDA_CHECK(cudaMemcpyAsync(&world_value, device_value, sizeof(float), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  const int color = global_rank % 2;
  const int key = world_size - global_rank;
  ncclConfig_t split_config = NCCL_CONFIG_INITIALIZER;
  ncclComm_t split_comm = nullptr;
  NCCL_CHECK(ncclCommSplit(world_comm, color, key, &split_comm, &split_config));

  int split_count = -1;
  int split_rank = -1;
  int split_device = -1;
  NCCL_CHECK(ncclCommCount(split_comm, &split_count));
  NCCL_CHECK(ncclCommUserRank(split_comm, &split_rank));
  NCCL_CHECK(ncclCommCuDevice(split_comm, &split_device));

  float split_value = static_cast<float>(global_rank + 1);
  CUDA_CHECK(cudaMemcpyAsync(device_value, &split_value, sizeof(float), cudaMemcpyHostToDevice, stream));
  NCCL_CHECK(ncclAllReduce(device_value, device_value, 1, ncclFloat, ncclSum, split_comm, stream));
  CUDA_CHECK(cudaMemcpyAsync(&split_value, device_value, sizeof(float), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  const float expected_world = 10.0f;
  const float expected_split = color == 0 ? 4.0f : 6.0f;
  const bool correct = world_value == expected_world && split_value == expected_split;

  std::ostringstream output_path;
  output_path << argv[1] << "/rank" << global_rank << ".csv";
  std::ofstream output(output_path.str());
  output << "pid,global_rank,world_size,cuda_device,pci_bus_id,unique_id_hash,"
            "comm_count,comm_rank,comm_device,color,key,split_count,split_rank,split_device,"
            "world_sum,split_sum,init_ms,correct\n";
  output << getpid() << ',' << global_rank << ',' << world_size << ',' << cuda_device << ','
         << pci_bus_id << ",0x" << std::hex << unique_id_hash << std::dec << ','
         << comm_count << ',' << comm_rank << ',' << comm_device << ','
         << color << ',' << key << ',' << split_count << ',' << split_rank << ',' << split_device << ','
         << std::fixed << std::setprecision(3) << world_value << ',' << split_value << ','
         << init_ms << ',' << (correct ? "true" : "false") << '\n';
  output.close();

  std::cout << "global_rank=" << global_rank << " cuda_device=" << cuda_device
            << " pci=" << pci_bus_id << " comm_rank=" << comm_rank
            << " color=" << color << " key=" << key << " split_rank=" << split_rank
            << " world_sum=" << world_value << " split_sum=" << split_value
            << " correct=" << correct << std::endl;

  MPI_Barrier(MPI_COMM_WORLD);
  NCCL_CHECK(ncclCommDestroy(split_comm));
  NCCL_CHECK(ncclCommDestroy(world_comm));
  CUDA_CHECK(cudaFree(device_value));
  CUDA_CHECK(cudaStreamDestroy(stream));
  MPI_Finalize();
  return correct ? 0 : 4;
}
