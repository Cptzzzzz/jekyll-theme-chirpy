#include <cuda.h>
#include <cuda_runtime.h>
#include <nccl.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CUDA_CHECK(x) do { auto e=(x); if(e!=cudaSuccess){std::fprintf(stderr,"CUDA %s\n",cudaGetErrorString(e));return 2;} } while(0)
#define NCCL_CHECK(x) do { auto e=(x); if(e!=ncclSuccess){std::fprintf(stderr,"NCCL %s\n",ncclGetErrorString(e));return 3;} } while(0)

static int attrs() {
  int n=0; CUDA_CHECK(cudaGetDeviceCount(&n));
  for(int d=0;d<n;d++) {
    int gdr=-1,dmabuf=-1;
    CUDA_CHECK(cudaDeviceGetAttribute(&gdr,cudaDevAttrGPUDirectRDMASupported,d));
#if CUDART_VERSION >= 11070
    CUdevice dev; if(cuDeviceGet(&dev,d)==CUDA_SUCCESS)
      cuDeviceGetAttribute(&dmabuf,CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED,dev);
#endif
    std::printf("CUDA_ATTR gpu=%d gdr=%d dmabuf=%d\n",d,gdr,dmabuf);
  }
  return 0;
}

static int init(std::vector<ncclComm_t>& comms) {
  comms.resize(4); NCCL_CHECK(ncclCommInitAll(comms.data(),4,nullptr)); return 0;
}

static int cache_case() {
  std::vector<ncclComm_t> comms; if(init(comms)) return 3;
  CUDA_CHECK(cudaSetDevice(0)); char* p=nullptr; CUDA_CHECK(cudaMalloc(&p,8<<20));
  void *h1=nullptr,*h2=nullptr,*hsub=nullptr,*hdis=nullptr;
  NCCL_CHECK(ncclCommRegister(comms[0],p,2<<20,&h1));
  NCCL_CHECK(ncclCommRegister(comms[0],p,2<<20,&h2));
  NCCL_CHECK(ncclCommRegister(comms[0],p+4096,1<<20,&hsub));
  NCCL_CHECK(ncclCommRegister(comms[0],p+(4<<20),2<<20,&hdis));
  std::printf("CACHE exact_same=%d contained_same=%d disjoint_distinct=%d handles_nonnull=%d\n",
    h1==h2,h1==hsub,h1!=hdis,h1&&h2&&hsub&&hdis);
  NCCL_CHECK(ncclCommDeregister(comms[0],h1));
  NCCL_CHECK(ncclCommDeregister(comms[0],h2));
  NCCL_CHECK(ncclCommDeregister(comms[0],hsub));
  ncclResult_t stale=ncclCommDeregister(comms[0],hsub);
  std::printf("CACHE stale_result=%d stale_is_invalid_usage=%d\n",(int)stale,stale==ncclInvalidUsage);
  NCCL_CHECK(ncclCommDeregister(comms[0],hdis));
  CUDA_CHECK(cudaFree(p)); for(auto c:comms) NCCL_CHECK(ncclCommDestroy(c)); return 0;
}

static int timings(int samples) {
  std::vector<ncclComm_t> comms; if(init(comms)) return 3;
  CUDA_CHECK(cudaSetDevice(0));
  for(size_t bytes: {size_t(4096),size_t(1<<20),size_t(64<<20)}) {
    char* p=nullptr; CUDA_CHECK(cudaMalloc(&p,bytes));
    for(int s=0;s<samples;s++) {
      void* h=nullptr; auto a=std::chrono::steady_clock::now();
      NCCL_CHECK(ncclCommRegister(comms[0],p,bytes,&h)); auto b=std::chrono::steady_clock::now();
      NCCL_CHECK(ncclCommDeregister(comms[0],h)); auto c=std::chrono::steady_clock::now();
      std::printf("TIMING bytes=%zu sample=%d reg_us=%.3f dereg_us=%.3f handle_nonnull=%d\n",bytes,s,
        std::chrono::duration<double,std::micro>(b-a).count(),std::chrono::duration<double,std::micro>(c-b).count(),h!=nullptr);
    }
    CUDA_CHECK(cudaFree(p));
  }
  for(auto c:comms) NCCL_CHECK(ncclCommDestroy(c)); return 0;
}

static int disabled_case() {
  std::vector<ncclComm_t> comms; if(init(comms)) return 3;
  CUDA_CHECK(cudaSetDevice(0)); void* p=nullptr;CUDA_CHECK(cudaMalloc(&p,4096));void* h=nullptr;
  ncclResult_t r=ncclCommRegister(comms[0],p,4096,&h);
  std::printf("DISABLED result=%d handle_null=%d\n",(int)r,h==nullptr);
  CUDA_CHECK(cudaFree(p));for(auto c:comms)NCCL_CHECK(ncclCommDestroy(c));return (r==ncclSuccess&&h==nullptr)?0:5;
}

int main(int argc,char**argv){
  const char* mode=argc>1?argv[1]:"attrs";int samples=argc>2?std::atoi(argv[2]):20;
  if(!std::strcmp(mode,"attrs"))return attrs();
  if(!std::strcmp(mode,"cache"))return cache_case();
  if(!std::strcmp(mode,"timing"))return timings(samples);
  if(!std::strcmp(mode,"disabled"))return disabled_case();return 4;
}
