#include <cuda_runtime.h>
#include <nccl.h>
#include <nvToolsExt.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CUDA_CHECK(cmd) do { cudaError_t e=(cmd); if(e!=cudaSuccess){ \
  std::fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
  std::exit(2); }} while(0)
#define NCCL_CHECK(cmd) do { ncclResult_t e=(cmd); if(e!=ncclSuccess){ \
  std::fprintf(stderr,"NCCL %s:%d %s\n",__FILE__,__LINE__,ncclGetErrorString(e)); \
  std::exit(3); }} while(0)

struct State {
  int nranks=4, ops=1;
  size_t count=0;
  std::vector<ncclComm_t> comms;
  std::vector<cudaStream_t> streams;
  std::vector<float*> send;
  std::vector<float*> recv;
};

static void sync_all(State& s) {
  for (int r=0;r<s.nranks;r++) {
    CUDA_CHECK(cudaSetDevice(r));
    CUDA_CHECK(cudaStreamSynchronize(s.streams[r]));
  }
}

static void enqueue_group(State& s, int ops) {
  NCCL_CHECK(ncclGroupStart());
  for (int op=0;op<ops;op++) {
    for (int r=0;r<s.nranks;r++) {
      CUDA_CHECK(cudaSetDevice(r));
      size_t i=(size_t)op*s.nranks+r;
      NCCL_CHECK(ncclAllReduce(s.send[i],s.recv[i],s.count,ncclFloat,
                              ncclSum,s.comms[r],s.streams[r]));
    }
  }
  NCCL_CHECK(ncclGroupEnd());
}

static void enqueue_sequential(State& s, int ops) {
  for (int op=0;op<ops;op++) {
    NCCL_CHECK(ncclGroupStart());
    for (int r=0;r<s.nranks;r++) {
      CUDA_CHECK(cudaSetDevice(r));
      size_t i=(size_t)op*s.nranks+r;
      NCCL_CHECK(ncclAllReduce(s.send[i],s.recv[i],s.count,ncclFloat,
                              ncclSum,s.comms[r],s.streams[r]));
    }
    NCCL_CHECK(ncclGroupEnd());
  }
}

static int verify(State& s, int ops) {
  const float expected=(float)(s.nranks*(s.nranks+1)/2);
  for (int op=0;op<ops;op++) for (int r=0;r<s.nranks;r++) {
    float edge[2]={};
    size_t i=(size_t)op*s.nranks+r;
    CUDA_CHECK(cudaSetDevice(r));
    CUDA_CHECK(cudaMemcpy(&edge[0],s.recv[i],sizeof(float),cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&edge[1],s.recv[i]+s.count-1,sizeof(float),cudaMemcpyDeviceToHost));
    if (std::fabs(edge[0]-expected)>1e-5 || std::fabs(edge[1]-expected)>1e-5)
      return 0;
  }
  return 1;
}

int main(int argc,char** argv) {
  std::string mode="grouped";
  int ops=4,replays=5;
  size_t bytes=4<<20;
  for(int i=1;i<argc;i++) {
    if(!std::strcmp(argv[i],"--mode") && ++i<argc) mode=argv[i];
    else if(!std::strcmp(argv[i],"--ops") && ++i<argc) ops=std::atoi(argv[i]);
    else if(!std::strcmp(argv[i],"--bytes") && ++i<argc) bytes=std::strtoull(argv[i],nullptr,0);
    else if(!std::strcmp(argv[i],"--replays") && ++i<argc) replays=std::atoi(argv[i]);
  }
  State s; s.ops=ops;s.count=bytes/sizeof(float);
  s.comms.resize(s.nranks);s.streams.resize(s.nranks);
  s.send.resize((size_t)ops*s.nranks);s.recv.resize((size_t)ops*s.nranks);
  NCCL_CHECK(ncclCommInitAll(s.comms.data(),s.nranks,nullptr));
  std::vector<float> host(s.count);
  for(int op=0;op<ops;op++) for(int r=0;r<s.nranks;r++) {
    CUDA_CHECK(cudaSetDevice(r));
    if(op==0) CUDA_CHECK(cudaStreamCreateWithFlags(&s.streams[r],cudaStreamNonBlocking));
    std::fill(host.begin(),host.end(),(float)(r+1));
    size_t i=(size_t)op*s.nranks+r;
    CUDA_CHECK(cudaMalloc(&s.send[i],bytes));CUDA_CHECK(cudaMalloc(&s.recv[i],bytes));
    CUDA_CHECK(cudaMemcpy(s.send[i],host.data(),bytes,cudaMemcpyHostToDevice));
  }
  enqueue_group(s,1);sync_all(s);
  for(int op=0;op<ops;op++) for(int r=0;r<s.nranks;r++) {
    size_t i=(size_t)op*s.nranks+r;CUDA_CHECK(cudaSetDevice(r));
    CUDA_CHECK(cudaMemsetAsync(s.recv[i],0,bytes,s.streams[r]));
  }
  sync_all(s);

  int graphNodes=0;
  auto begin=std::chrono::steady_clock::now();
  nvtxRangePushA((mode+"-ops"+std::to_string(ops)).c_str());
  if(mode=="sequential") {
    enqueue_sequential(s,ops);
  } else if(mode=="grouped") {
    enqueue_group(s,ops);
  } else if(mode=="graph") {
    std::vector<cudaGraph_t> graphs(s.nranks);
    std::vector<cudaGraphExec_t> execs(s.nranks);
    for(int r=0;r<s.nranks;r++) {
      CUDA_CHECK(cudaSetDevice(r));
      CUDA_CHECK(cudaStreamBeginCapture(s.streams[r],cudaStreamCaptureModeRelaxed));
    }
    enqueue_group(s,ops);
    for(int r=0;r<s.nranks;r++) {
      CUDA_CHECK(cudaSetDevice(r));CUDA_CHECK(cudaStreamEndCapture(s.streams[r],&graphs[r]));
      size_t n=0;CUDA_CHECK(cudaGraphGetNodes(graphs[r],nullptr,&n));graphNodes+=(int)n;
      CUDA_CHECK(cudaGraphInstantiate(&execs[r],graphs[r],nullptr,nullptr,0));
    }
    for(int k=0;k<replays;k++) for(int r=0;r<s.nranks;r++) {
      CUDA_CHECK(cudaSetDevice(r));CUDA_CHECK(cudaGraphLaunch(execs[r],s.streams[r]));
    }
    for(int r=0;r<s.nranks;r++) {
      CUDA_CHECK(cudaSetDevice(r));CUDA_CHECK(cudaGraphExecDestroy(execs[r]));
      CUDA_CHECK(cudaGraphDestroy(graphs[r]));
    }
  } else {
    std::fprintf(stderr,"bad mode\n");return 4;
  }
  auto submitted=std::chrono::steady_clock::now();
  sync_all(s);
  auto done=std::chrono::steady_clock::now();
  nvtxRangePop();
  int ok=verify(s,ops);
  double hostUs=std::chrono::duration<double,std::micro>(submitted-begin).count();
  double totalUs=std::chrono::duration<double,std::micro>(done-begin).count();
  std::printf("RESULT mode=%s ops=%d replays=%d bytes=%zu host_us=%.3f total_us=%.3f graph_nodes=%d correct=%d\n",
              mode.c_str(),ops,mode=="graph"?replays:1,bytes,hostUs,totalUs,graphNodes,ok);

  for(int op=0;op<ops;op++) for(int r=0;r<s.nranks;r++) {
    size_t i=(size_t)op*s.nranks+r;CUDA_CHECK(cudaSetDevice(r));
    CUDA_CHECK(cudaFree(s.send[i]));CUDA_CHECK(cudaFree(s.recv[i]));
  }
  for(int r=0;r<s.nranks;r++) {
    CUDA_CHECK(cudaSetDevice(r));CUDA_CHECK(cudaStreamDestroy(s.streams[r]));
    NCCL_CHECK(ncclCommDestroy(s.comms[r]));
  }
  return ok?0:5;
}
