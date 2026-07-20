#include <cuda.h>
#include <cstdio>
int main(){
  if(cuInit(0)!=CUDA_SUCCESS)return 2;int n=0;if(cuDeviceGetCount(&n)!=CUDA_SUCCESS)return 3;
  for(int i=0;i<n;i++){CUdevice d;int mc=-1,ccmaj=-1,ccmin=-1;cuDeviceGet(&d,i);
    cuDeviceGetAttribute(&mc,CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED,d);
    cuDeviceGetAttribute(&ccmaj,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,d);
    cuDeviceGetAttribute(&ccmin,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,d);
    std::printf("NVLS_ATTR gpu=%d multicast=%d cc=%d%d\n",i,mc,ccmaj,ccmin);}
  return 0;
}
