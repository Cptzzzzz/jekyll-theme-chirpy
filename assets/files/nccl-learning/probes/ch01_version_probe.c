#define _GNU_SOURCE

#include <dlfcn.h>
#include <nccl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

static void decode_version(int code, int* major, int* minor, int* patch) {
  if (code < 10000) {
    *major = code / 1000;
    *minor = (code % 1000) / 100;
    *patch = code % 100;
  } else {
    *major = code / 10000;
    *minor = (code % 10000) / 100;
    *patch = code % 100;
  }
}

int main(int argc, char** argv) {
  int runtime_code = 0;
  int major = 0;
  int minor = 0;
  int patch = 0;
  int hold_seconds = argc > 1 ? atoi(argv[1]) : 0;
  Dl_info symbol_info = {0};

  ncclResult_t result = ncclGetVersion(&runtime_code);
  if (result != ncclSuccess) {
    fprintf(stderr, "ncclGetVersion failed: result=%d\n", (int)result);
    return 1;
  }

  decode_version(runtime_code, &major, &minor, &patch);
  dladdr((void*)&ncclGetVersion, &symbol_info);

  printf("process_id=%d\n", (int)getpid());
  printf("header_version_code=%d\n", NCCL_VERSION_CODE);
  printf("header_version=%d.%d.%d%s\n", NCCL_MAJOR, NCCL_MINOR, NCCL_PATCH, NCCL_SUFFIX);
  printf("runtime_version_code=%d\n", runtime_code);
  printf("runtime_version=%d.%d.%d\n", major, minor, patch);
  printf("nccl_get_version_object=%s\n", symbol_info.dli_fname ? symbol_info.dli_fname : "unknown");
  printf("hold_seconds=%d\n", hold_seconds);
  fflush(stdout);

  if (hold_seconds > 0) sleep((unsigned int)hold_seconds);
  return 0;
}
