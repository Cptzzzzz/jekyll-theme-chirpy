#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    cudaError_t status_ = (call);                                                \
    if (status_ != cudaSuccess) {                                                \
      std::cerr << "CUDA failure " << __FILE__ << ':' << __LINE__ << ": "      \
                << cudaGetErrorString(status_) << std::endl;                    \
      MPI_Abort(MPI_COMM_WORLD, 10);                                             \
    }                                                                           \
  } while (0)

#define NCCL_CHECK(call)                                                        \
  do {                                                                          \
    ncclResult_t status_ = (call);                                               \
    if (status_ != ncclSuccess) {                                                \
      std::cerr << "NCCL failure " << __FILE__ << ':' << __LINE__ << ": "      \
                << ncclGetErrorString(status_) << std::endl;                    \
      MPI_Abort(MPI_COMM_WORLD, 11);                                             \
    }                                                                           \
  } while (0)

static std::string encode(const std::vector<int>& values) {
  std::ostringstream output;
  for (size_t index = 0; index < values.size(); ++index) {
    if (index != 0) output << ';';
    output << values[index];
  }
  return output.str();
}

struct ResultRow {
  std::string operation;
  std::string variant;
  std::string actual;
  std::string expected;
  std::string pointer_relation;
  bool correct;
};

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0;
  int world = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world);
  if (world != 4 || argc != 2) {
    if (rank == 0) std::cerr << "usage: mpirun -np 4 program OUTPUT_DIR\n";
    MPI_Abort(MPI_COMM_WORLD, 2);
  }

  CUDA_CHECK(cudaSetDevice(rank));
  ncclUniqueId id{};
  if (rank == 0) NCCL_CHECK(ncclGetUniqueId(&id));
  MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

  ncclComm_t comm = nullptr;
  NCCL_CHECK(ncclCommInitRank(&comm, world, id, rank));
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  constexpr size_t kCapacity = 32;
  int* send = nullptr;
  int* recv = nullptr;
  CUDA_CHECK(cudaMalloc(&send, kCapacity * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&recv, kCapacity * sizeof(int)));

  auto upload = [&](int* device, const std::vector<int>& host) {
    CUDA_CHECK(cudaMemcpyAsync(device, host.data(), host.size() * sizeof(int),
                               cudaMemcpyHostToDevice, stream));
  };
  auto download = [&](const int* device, size_t count) {
    std::vector<int> host(count);
    CUDA_CHECK(cudaMemcpyAsync(host.data(), device, count * sizeof(int),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return host;
  };

  std::vector<ResultRow> rows;
  auto record = [&](const std::string& operation, const std::string& variant,
                    const std::vector<int>& actual,
                    const std::vector<int>& expected,
                    const std::string& pointer_relation) {
    rows.push_back({operation, variant, encode(actual), encode(expected),
                    pointer_relation, actual == expected});
  };

  // AllReduce: ordinary out-of-place buffers.
  std::vector<int> ar_input = {rank + 1, rank + 10, rank * 2};
  const std::vector<int> ar_expected = {10, 46, 12};
  upload(send, ar_input);
  NCCL_CHECK(ncclAllReduce(send, recv, ar_input.size(), ncclInt32, ncclSum,
                           comm, stream));
  record("all_reduce", "out-of-place", download(recv, ar_input.size()),
         ar_expected, "sendbuff!=recvbuff");

  // AllReduce: exact pointer equality activates the in-place contract.
  upload(send, ar_input);
  NCCL_CHECK(ncclAllReduce(send, send, ar_input.size(), ncclInt32, ncclSum,
                           comm, stream));
  record("all_reduce", "in-place", download(send, ar_input.size()),
         ar_expected, "sendbuff==recvbuff");

  const size_t gather_count = 3;
  std::vector<int> gather_input = {rank, rank + 10, rank + 100};
  std::vector<int> gather_expected;
  for (int source = 0; source < world; ++source) {
    gather_expected.insert(gather_expected.end(),
                           {source, source + 10, source + 100});
  }
  upload(send, gather_input);
  NCCL_CHECK(ncclAllGather(send, recv, gather_count, ncclInt32, comm, stream));
  record("all_gather", "out-of-place", download(recv, world * gather_count),
         gather_expected, "sendbuff separate;recvbuff[nranks*sendcount]");

  // In-place AllGather does not use recvbuff itself as sendbuff. Each rank's
  // source starts at its own rank-major slice inside the receive buffer.
  std::vector<int> gather_storage(world * gather_count, -1);
  std::copy(gather_input.begin(), gather_input.end(),
            gather_storage.begin() + rank * gather_count);
  upload(recv, gather_storage);
  NCCL_CHECK(ncclAllGather(recv + rank * gather_count, recv, gather_count,
                           ncclInt32, comm, stream));
  record("all_gather", "in-place-rank-offset",
         download(recv, world * gather_count), gather_expected,
         "sendbuff==recvbuff+rank*sendcount");

  const size_t scatter_count = 3;
  std::vector<int> scatter_input(world * scatter_count);
  for (size_t index = 0; index < scatter_input.size(); ++index) {
    scatter_input[index] = rank * 100 + static_cast<int>(index);
  }
  std::vector<int> scatter_expected(scatter_count);
  for (size_t index = 0; index < scatter_count; ++index) {
    const int global_index = rank * scatter_count + index;
    scatter_expected[index] = 600 + world * global_index;
  }
  upload(send, scatter_input);
  NCCL_CHECK(ncclReduceScatter(send, recv, scatter_count, ncclInt32, ncclSum,
                               comm, stream));
  record("reduce_scatter", "out-of-place", download(recv, scatter_count),
         scatter_expected, "sendbuff[nranks*recvcount];recvbuff separate");

  // In-place ReduceScatter writes this rank's reduced shard back into the
  // matching rank slice of the input allocation.
  upload(send, scatter_input);
  NCCL_CHECK(ncclReduceScatter(send, send + rank * scatter_count, scatter_count,
                               ncclInt32, ncclSum, comm, stream));
  record("reduce_scatter", "in-place-rank-offset",
         download(send + rank * scatter_count, scatter_count), scatter_expected,
         "recvbuff==sendbuff+rank*recvcount");

  std::vector<int> broadcast_input = {rank * 10, rank * 10 + 1,
                                      rank * 10 + 2, rank * 10 + 3};
  const std::vector<int> broadcast_root2 = {20, 21, 22, 23};
  upload(send, broadcast_input);
  NCCL_CHECK(ncclBroadcast(send, recv, broadcast_input.size(), ncclInt32, 2,
                           comm, stream));
  record("broadcast", "out-of-place-root-2",
         download(recv, broadcast_input.size()), broadcast_root2,
         "sendbuff!=recvbuff;root is communicator rank");

  const std::vector<int> broadcast_root1 = {10, 11, 12, 13};
  upload(send, broadcast_input);
  NCCL_CHECK(ncclBroadcast(send, send, broadcast_input.size(), ncclInt32, 1,
                           comm, stream));
  record("broadcast", "in-place-root-1",
         download(send, broadcast_input.size()), broadcast_root1,
         "sendbuff==recvbuff");

  std::vector<int> reduce_input = {rank + 1, (rank + 1) * 10};
  const std::vector<int> reduce_expected = {10, 100};
  upload(send, reduce_input);
  void* reduce_output = rank == 3 ? static_cast<void*>(recv) : nullptr;
  NCCL_CHECK(ncclReduce(send, reduce_output, reduce_input.size(), ncclInt32,
                        ncclSum, 3, comm, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  if (rank == 3) {
    record("reduce", "out-of-place-root-3", download(recv, reduce_input.size()),
           reduce_expected, "root recvbuff valid");
  } else {
    rows.push_back({"reduce", "out-of-place-root-3", "undefined", "undefined",
                    "non-root recvbuff==nullptr", true});
  }

  std::ostringstream path;
  path << argv[1] << "/rank" << rank << ".csv";
  std::ofstream output(path.str());
  output << "rank,operation,variant,actual,expected,pointer_relation,correct\n";
  for (const auto& row : rows) {
    output << rank << ',' << row.operation << ',' << row.variant << ','
           << row.actual << ',' << row.expected << ',' << row.pointer_relation
           << ',' << (row.correct ? "true" : "false") << '\n';
  }
  output.close();

  bool correct = true;
  for (const auto& row : rows) correct = correct && row.correct;
  std::cout << "rank=" << rank << " native_variants=" << rows.size()
            << " correct=" << correct << std::endl;

  MPI_Barrier(MPI_COMM_WORLD);
  CUDA_CHECK(cudaFree(recv));
  CUDA_CHECK(cudaFree(send));
  CUDA_CHECK(cudaStreamDestroy(stream));
  NCCL_CHECK(ncclCommDestroy(comm));
  MPI_Finalize();
  return correct ? 0 : 4;
}
