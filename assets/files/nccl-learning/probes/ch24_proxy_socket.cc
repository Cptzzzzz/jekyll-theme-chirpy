#include <cuda_runtime.h>
#include <nccl.h>
#include <nvToolsExt.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

#define CUDA_CHECK(cmd) do { cudaError_t e=(cmd); if(e!=cudaSuccess){ \
  std::fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
  std::exit(2); }} while(0)
#define NCCL_CHECK(cmd) do { ncclResult_t e=(cmd); if(e!=ncclSuccess){ \
  std::fprintf(stderr,"NCCL %s:%d %s\n",__FILE__,__LINE__,ncclGetErrorString(e)); \
  std::exit(3); }} while(0)

static void enqueue(std::vector<ncclComm_t>& comms,
                    std::vector<cudaStream_t>& streams,
                    std::vector<float*>& send,
                    std::vector<float*>& recv, size_t count) {
  NCCL_CHECK(ncclGroupStart());
  for (int rank=0; rank<(int)comms.size(); rank++) {
    CUDA_CHECK(cudaSetDevice(rank));
    NCCL_CHECK(ncclAllReduce(send[rank], recv[rank], count, ncclFloat,
                            ncclSum, comms[rank], streams[rank]));
  }
  NCCL_CHECK(ncclGroupEnd());
}

static void sync_all(std::vector<cudaStream_t>& streams) {
  for (int rank=0; rank<(int)streams.size(); rank++) {
    CUDA_CHECK(cudaSetDevice(rank));
    CUDA_CHECK(cudaStreamSynchronize(streams[rank]));
  }
}

int main(int argc, char** argv) {
  int nranks=4, iterations=10, pauseMs=0, donePauseMs=0;
  size_t bytes=64ull<<20;
  for (int i=1; i<argc; i++) {
    if (!std::strcmp(argv[i],"--iterations") && ++i<argc) iterations=std::atoi(argv[i]);
    else if (!std::strcmp(argv[i],"--bytes") && ++i<argc) bytes=std::strtoull(argv[i],nullptr,0);
    else if (!std::strcmp(argv[i],"--pause-ms") && ++i<argc) pauseMs=std::atoi(argv[i]);
    else if (!std::strcmp(argv[i],"--done-pause-ms") && ++i<argc) donePauseMs=std::atoi(argv[i]);
    else if (!std::strcmp(argv[i],"--ranks") && ++i<argc) nranks=std::atoi(argv[i]);
  }
  if (nranks<2 || bytes<sizeof(float) || bytes%sizeof(float) || iterations<1) return 4;

  const size_t count=bytes/sizeof(float);
  std::vector<ncclComm_t> comms(nranks);
  std::vector<cudaStream_t> streams(nranks);
  std::vector<float*> send(nranks),recv(nranks);
  std::vector<float> host(count);
  NCCL_CHECK(ncclCommInitAll(comms.data(),nranks,nullptr));
  for (int rank=0; rank<nranks; rank++) {
    CUDA_CHECK(cudaSetDevice(rank));
    CUDA_CHECK(cudaStreamCreateWithFlags(&streams[rank],cudaStreamNonBlocking));
    CUDA_CHECK(cudaMalloc(&send[rank],bytes));
    CUDA_CHECK(cudaMalloc(&recv[rank],bytes));
    std::fill(host.begin(),host.end(),(float)(rank+1));
    CUDA_CHECK(cudaMemcpy(send[rank],host.data(),bytes,cudaMemcpyHostToDevice));
  }

  enqueue(comms,streams,send,recv,count);
  sync_all(streams);
  std::printf("READY pid=%d ranks=%d bytes=%zu iterations=%d\n",
              (int)getpid(),nranks,bytes,iterations);
  std::fflush(stdout);
  if (pauseMs) std::this_thread::sleep_for(std::chrono::milliseconds(pauseMs));

  auto begin=std::chrono::steady_clock::now();
  nvtxRangePushA("net-loop");
  for (int i=0; i<iterations; i++) enqueue(comms,streams,send,recv,count);
  auto submitted=std::chrono::steady_clock::now();
  sync_all(streams);
  nvtxRangePop();
  auto done=std::chrono::steady_clock::now();

  const float expected=(float)(nranks*(nranks+1)/2);
  int correct=1;
  for (int rank=0; rank<nranks; rank++) {
    float edge[2]={};
    CUDA_CHECK(cudaSetDevice(rank));
    CUDA_CHECK(cudaMemcpy(&edge[0],recv[rank],sizeof(float),cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&edge[1],recv[rank]+count-1,sizeof(float),cudaMemcpyDeviceToHost));
    if (std::fabs(edge[0]-expected)>1e-5 || std::fabs(edge[1]-expected)>1e-5) correct=0;
  }
  double hostUs=std::chrono::duration<double,std::micro>(submitted-begin).count();
  double totalUs=std::chrono::duration<double,std::micro>(done-begin).count();
  std::printf("RESULT iterations=%d bytes=%zu host_us=%.3f total_us=%.3f correct=%d\n",
              iterations,bytes,hostUs,totalUs,correct);
  std::fflush(stdout);
  if (donePauseMs) std::this_thread::sleep_for(std::chrono::milliseconds(donePauseMs));

  for (int rank=0; rank<nranks; rank++) {
    CUDA_CHECK(cudaSetDevice(rank));
    CUDA_CHECK(cudaFree(send[rank]));
    CUDA_CHECK(cudaFree(recv[rank]));
    CUDA_CHECK(cudaStreamDestroy(streams[rank]));
    NCCL_CHECK(ncclCommDestroy(comms[rank]));
  }
  return correct ? 0 : 5;
}
