#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
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

using Clock = std::chrono::steady_clock;

static double elapsed_us(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration<double, std::micro>(end - start).count();
}

struct Row {
  std::string kind;
  std::string variant;
  int cycle;
  double host_us;
  double device_us;
  int event_ready_after_group_end;
  bool correct;
  std::string detail;
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

  std::vector<Row> rows;
  auto record_semantic = [&](const std::string& variant, bool correct,
                             const std::string& detail, double host_us = 0.0) {
    rows.push_back({"semantic", variant, -1, host_us, 0.0, -1, correct, detail});
  };

  // This error is thread-local and does not require a communicator.
  const ncclResult_t end_without_start = ncclGroupEnd();
  record_semantic("group-end-without-start",
                  end_without_start == ncclInvalidUsage,
                  std::string("result=") + ncclGetErrorString(end_without_start));

  ncclUniqueId id{};
  if (rank == 0) NCCL_CHECK(ncclGetUniqueId(&id));
  MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
  ncclComm_t comm = nullptr;
  NCCL_CHECK(ncclCommInitRank(&comm, world, id, rank));

  cudaStream_t stream_a = nullptr;
  cudaStream_t stream_b = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream_a, cudaStreamNonBlocking));
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream_b, cudaStreamNonBlocking));

  constexpr size_t kSmallCount = 4;
  int* small_a = nullptr;
  int* small_b = nullptr;
  int* small_c = nullptr;
  CUDA_CHECK(cudaMalloc(&small_a, kSmallCount * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&small_b, kSmallCount * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&small_c, kSmallCount * sizeof(int)));

  auto upload = [&](int* device, const std::vector<int>& host, cudaStream_t stream) {
    CUDA_CHECK(cudaMemcpyAsync(device, host.data(), host.size() * sizeof(int),
                               cudaMemcpyHostToDevice, stream));
  };
  auto download = [&](int* device, size_t count, cudaStream_t stream) {
    std::vector<int> host(count);
    CUDA_CHECK(cudaMemcpyAsync(host.data(), device, count * sizeof(int),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return host;
  };

  const std::vector<int> expected_a = {10, 14, 18, 22};
  const std::vector<int> expected_b = {46, 50, 54, 58};
  const std::vector<int> expected_c = {86, 90, 94, 98};
  upload(small_a, {rank + 1, rank + 2, rank + 3, rank + 4}, stream_a);
  upload(small_b, {rank + 10, rank + 11, rank + 12, rank + 13}, stream_a);
  upload(small_c, {rank + 20, rank + 21, rank + 22, rank + 23}, stream_a);
  CUDA_CHECK(cudaStreamSynchronize(stream_a));

  // Inner GroupEnd only decrements nesting depth. Launch happens at outer end.
  auto nested_start = Clock::now();
  NCCL_CHECK(ncclGroupStart());
  NCCL_CHECK(ncclAllReduce(small_a, small_a, kSmallCount, ncclInt32,
                           ncclSum, comm, stream_a));
  NCCL_CHECK(ncclGroupStart());
  NCCL_CHECK(ncclAllReduce(small_b, small_b, kSmallCount, ncclInt32,
                           ncclSum, comm, stream_a));
  const ncclResult_t inner_end = ncclGroupEnd();
  NCCL_CHECK(ncclAllReduce(small_c, small_c, kSmallCount, ncclInt32,
                           ncclSum, comm, stream_a));
  const ncclResult_t outer_end = ncclGroupEnd();
  auto nested_end = Clock::now();
  CUDA_CHECK(cudaStreamSynchronize(stream_a));
  const bool nested_correct =
      inner_end == ncclSuccess && outer_end == ncclSuccess &&
      download(small_a, kSmallCount, stream_a) == expected_a &&
      download(small_b, kSmallCount, stream_a) == expected_b &&
      download(small_c, kSmallCount, stream_a) == expected_c;
  record_semantic("nested-depth-two", nested_correct,
                  "three ordered allreduces;inner and outer end success",
                  elapsed_us(nested_start, nested_end));

  // One outer group can contain tasks on two user streams.
  upload(small_a, {rank + 1, rank + 2, rank + 3, rank + 4}, stream_a);
  upload(small_b, {rank + 10, rank + 11, rank + 12, rank + 13}, stream_b);
  CUDA_CHECK(cudaStreamSynchronize(stream_a));
  CUDA_CHECK(cudaStreamSynchronize(stream_b));
  NCCL_CHECK(ncclGroupStart());
  NCCL_CHECK(ncclAllReduce(small_a, small_a, kSmallCount, ncclInt32,
                           ncclSum, comm, stream_a));
  NCCL_CHECK(ncclAllReduce(small_b, small_b, kSmallCount, ncclInt32,
                           ncclSum, comm, stream_b));
  NCCL_CHECK(ncclGroupEnd());
  CUDA_CHECK(cudaStreamSynchronize(stream_a));
  CUDA_CHECK(cudaStreamSynchronize(stream_b));
  record_semantic("two-user-streams",
                  download(small_a, kSmallCount, stream_a) == expected_a &&
                      download(small_b, kSmallCount, stream_b) == expected_b,
                  "one group;one allreduce per stream");

  ncclResult_t async_before = ncclInternalError;
  NCCL_CHECK(ncclCommGetAsyncError(comm, &async_before));
  record_semantic("async-error-after-success", async_before == ncclSuccess,
                  std::string("async=") + ncclGetErrorString(async_before));

  // An error inside a group is retained until the matching outer GroupEnd,
  // which must still be called to clean up thread-local group state.
  upload(small_a, {rank + 1, rank + 2, rank + 3, rank + 4}, stream_a);
  NCCL_CHECK(ncclGroupStart());
  const ncclResult_t invalid_call = ncclBroadcast(
      small_a, small_a, kSmallCount, ncclInt32, world, comm, stream_a);
  const ncclResult_t invalid_end = ncclGroupEnd();
  record_semantic(
      "invalid-root-group-cleanup",
      invalid_call == ncclInvalidArgument && invalid_end == ncclInvalidArgument,
      std::string("api=") + ncclGetErrorString(invalid_call) +
          ";group_end=" + ncclGetErrorString(invalid_end));

  // A blocking communicator remains usable after this local argument error and
  // balanced GroupEnd cleanup.
  upload(small_a, {rank + 1, rank + 2, rank + 3, rank + 4}, stream_a);
  NCCL_CHECK(ncclAllReduce(small_a, small_a, kSmallCount, ncclInt32,
                           ncclSum, comm, stream_a));
  const bool recovered = download(small_a, kSmallCount, stream_a) == expected_a;
  record_semantic("valid-op-after-group-error", recovered,
                  "balanced group cleanup restored subsequent use");

  // Exercise the public nonblocking communicator state machine. ncclInProgress
  // is polled through ncclCommGetAsyncError until it reaches a terminal state.
  auto poll_async = [&](ncclComm_t target, int* polls, double* poll_us) {
    ncclResult_t async = ncclInProgress;
    const auto poll_start = Clock::now();
    *polls = 0;
    while (async == ncclInProgress) {
      NCCL_CHECK(ncclCommGetAsyncError(target, &async));
      ++*polls;
      if (async == ncclInProgress) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
      if (elapsed_us(poll_start, Clock::now()) > 20.0 * 1000.0 * 1000.0) {
        return ncclSystemError;
      }
    }
    *poll_us = elapsed_us(poll_start, Clock::now());
    return async;
  };

  ncclUniqueId nonblocking_id{};
  if (rank == 0) NCCL_CHECK(ncclGetUniqueId(&nonblocking_id));
  MPI_Bcast(&nonblocking_id, sizeof(nonblocking_id), MPI_BYTE, 0, MPI_COMM_WORLD);
  ncclConfig_t nonblocking_config = NCCL_CONFIG_INITIALIZER;
  nonblocking_config.blocking = 0;
  ncclComm_t nonblocking_comm = nullptr;
  const ncclResult_t nonblocking_init = ncclCommInitRankConfig(
      &nonblocking_comm, world, nonblocking_id, rank, &nonblocking_config);
  int init_polls = 0;
  double init_poll_us = 0.0;
  const ncclResult_t init_terminal =
      poll_async(nonblocking_comm, &init_polls, &init_poll_us);
  record_semantic(
      "nonblocking-init-poll",
      (nonblocking_init == ncclInProgress || nonblocking_init == ncclSuccess) &&
          init_terminal == ncclSuccess,
      std::string("initial=") + ncclGetErrorString(nonblocking_init) +
          ";terminal=" + ncclGetErrorString(init_terminal) +
          ";polls=" + std::to_string(init_polls),
      init_poll_us);

  upload(small_c, {rank + 20, rank + 21, rank + 22, rank + 23}, stream_a);
  CUDA_CHECK(cudaStreamSynchronize(stream_a));
  NCCL_CHECK(ncclGroupStart());
  const ncclResult_t nonblocking_api = ncclAllReduce(
      small_c, small_c, kSmallCount, ncclInt32, ncclSum,
      nonblocking_comm, stream_a);
  const ncclResult_t nonblocking_group_end = ncclGroupEnd();
  int collective_polls = 0;
  double collective_poll_us = 0.0;
  const ncclResult_t collective_terminal =
      poll_async(nonblocking_comm, &collective_polls, &collective_poll_us);
  CUDA_CHECK(cudaStreamSynchronize(stream_a));
  const bool nonblocking_data =
      download(small_c, kSmallCount, stream_a) == expected_c;
  record_semantic(
      "nonblocking-group-poll",
      (nonblocking_api == ncclSuccess || nonblocking_api == ncclInProgress) &&
          (nonblocking_group_end == ncclSuccess ||
           nonblocking_group_end == ncclInProgress) &&
          collective_terminal == ncclSuccess && nonblocking_data,
      std::string("api=") + ncclGetErrorString(nonblocking_api) +
          ";group_end=" + ncclGetErrorString(nonblocking_group_end) +
          ";terminal=" + ncclGetErrorString(collective_terminal) +
          ";polls=" + std::to_string(collective_polls),
      collective_poll_us);

  const ncclResult_t finalize_initial = ncclCommFinalize(nonblocking_comm);
  int finalize_polls = 0;
  double finalize_poll_us = 0.0;
  const ncclResult_t finalize_terminal =
      poll_async(nonblocking_comm, &finalize_polls, &finalize_poll_us);
  const bool finalize_correct =
      (finalize_initial == ncclInProgress || finalize_initial == ncclSuccess) &&
      finalize_terminal == ncclSuccess;
  record_semantic(
      "nonblocking-finalize-poll", finalize_correct,
      std::string("initial=") + ncclGetErrorString(finalize_initial) +
          ";terminal=" + ncclGetErrorString(finalize_terminal) +
          ";polls=" + std::to_string(finalize_polls),
      finalize_poll_us);
  NCCL_CHECK(ncclCommDestroy(nonblocking_comm));

  // A large collective makes the enqueue/device-completion distinction visible.
  constexpr size_t kLargeBytes = 256ULL * 1024ULL * 1024ULL;
  constexpr size_t kLargeCount = kLargeBytes / sizeof(int);
  int* large = nullptr;
  CUDA_CHECK(cudaMalloc(&large, kLargeBytes));
  cudaEvent_t start_event = nullptr;
  cudaEvent_t stop_event = nullptr;
  CUDA_CHECK(cudaEventCreate(&start_event));
  CUDA_CHECK(cudaEventCreate(&stop_event));

  for (int warmup = 0; warmup < 3; ++warmup) {
    CUDA_CHECK(cudaMemsetAsync(large, rank + 1, kLargeBytes, stream_a));
    CUDA_CHECK(cudaStreamSynchronize(stream_a));
    MPI_Barrier(MPI_COMM_WORLD);
    NCCL_CHECK(ncclGroupStart());
    NCCL_CHECK(ncclAllReduce(large, large, kLargeCount, ncclInt32,
                             ncclSum, comm, stream_a));
    NCCL_CHECK(ncclGroupEnd());
    CUDA_CHECK(cudaStreamSynchronize(stream_a));
  }

  for (int cycle = 0; cycle < 10; ++cycle) {
    CUDA_CHECK(cudaMemsetAsync(large, rank + 1, kLargeBytes, stream_a));
    CUDA_CHECK(cudaStreamSynchronize(stream_a));
    MPI_Barrier(MPI_COMM_WORLD);
    CUDA_CHECK(cudaEventRecord(start_event, stream_a));
    auto host_start = Clock::now();
    NCCL_CHECK(ncclGroupStart());
    NCCL_CHECK(ncclAllReduce(large, large, kLargeCount, ncclInt32,
                             ncclSum, comm, stream_a));
    NCCL_CHECK(ncclGroupEnd());
    auto host_end = Clock::now();
    CUDA_CHECK(cudaEventRecord(stop_event, stream_a));
    const cudaError_t query = cudaEventQuery(stop_event);
    if (query != cudaSuccess && query != cudaErrorNotReady) CUDA_CHECK(query);
    CUDA_CHECK(cudaEventSynchronize(stop_event));
    float gpu_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, start_event, stop_event));
    int first = 0;
    int last = 0;
    CUDA_CHECK(cudaMemcpy(&first, large, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&last, large + kLargeCount - 1, sizeof(int),
                          cudaMemcpyDeviceToHost));
    const int expected = 0x0a0a0a0a;
    rows.push_back({"completion", "grouped-256MiB-allreduce", cycle,
                    elapsed_us(host_start, host_end), gpu_ms * 1000.0,
                    query == cudaSuccess ? 1 : 0,
                    first == expected && last == expected,
                    "host group interval versus CUDA event interval"});
  }

  CUDA_CHECK(cudaEventDestroy(stop_event));
  CUDA_CHECK(cudaEventDestroy(start_event));
  CUDA_CHECK(cudaFree(large));

  std::ostringstream output_path;
  output_path << argv[1] << "/rank" << rank << ".csv";
  std::ofstream output(output_path.str());
  output << "rank,kind,variant,cycle,host_us,device_us,event_ready_after_group_end,correct,detail\n";
  for (const auto& row : rows) {
    output << rank << ',' << row.kind << ',' << row.variant << ',' << row.cycle
           << ',' << row.host_us << ',' << row.device_us << ','
           << row.event_ready_after_group_end << ','
           << (row.correct ? "true" : "false") << ',' << row.detail << '\n';
  }
  output.close();

  bool all_correct = true;
  for (const auto& row : rows) all_correct = all_correct && row.correct;
  std::cout << "rank=" << rank << " rows=" << rows.size()
            << " all_correct=" << all_correct << std::endl;

  MPI_Barrier(MPI_COMM_WORLD);
  CUDA_CHECK(cudaFree(small_c));
  CUDA_CHECK(cudaFree(small_b));
  CUDA_CHECK(cudaFree(small_a));
  CUDA_CHECK(cudaStreamDestroy(stream_b));
  CUDA_CHECK(cudaStreamDestroy(stream_a));
  NCCL_CHECK(ncclCommDestroy(comm));
  MPI_Finalize();
  return all_correct ? 0 : 4;
}
