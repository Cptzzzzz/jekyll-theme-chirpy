#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

typedef ssize_t (*send_fn)(int,const void*,size_t,int);
typedef ssize_t (*recv_fn)(int,void*,size_t,int);
static send_fn real_send;
static recv_fn real_recv;
static _Atomic unsigned long long send_calls,recv_calls,send_bytes,recv_bytes,delayed_calls;
static unsigned long delay_us;
static size_t min_bytes=65536;

static void init_shim(void) __attribute__((constructor));
static void init_shim(void) {
  real_send=(send_fn)dlsym(RTLD_NEXT,"send");
  real_recv=(recv_fn)dlsym(RTLD_NEXT,"recv");
  const char* d=getenv("NCCL_SOCKET_DELAY_US");
  const char* m=getenv("NCCL_SOCKET_DELAY_MIN_BYTES");
  if (d) delay_us=strtoul(d,NULL,10);
  if (m) min_bytes=strtoull(m,NULL,10);
}

static void maybe_delay(size_t bytes) {
  if (!delay_us || bytes<min_bytes) return;
  struct timespec ts={(time_t)(delay_us/1000000),(long)(delay_us%1000000)*1000};
  atomic_fetch_add_explicit(&delayed_calls,1,memory_order_relaxed);
  syscall(SYS_nanosleep,&ts,NULL);
}

ssize_t send(int fd,const void* buf,size_t len,int flags) {
  atomic_fetch_add_explicit(&send_calls,1,memory_order_relaxed);
  ssize_t done=real_send(fd,buf,len,flags);
  if (done>0) {
    atomic_fetch_add_explicit(&send_bytes,(unsigned long long)done,memory_order_relaxed);
    maybe_delay((size_t)done);
  }
  return done;
}

ssize_t recv(int fd,void* buf,size_t len,int flags) {
  atomic_fetch_add_explicit(&recv_calls,1,memory_order_relaxed);
  ssize_t done=real_recv(fd,buf,len,flags);
  if (done>0) {
    atomic_fetch_add_explicit(&recv_bytes,(unsigned long long)done,memory_order_relaxed);
    maybe_delay((size_t)done);
  }
  return done;
}

static void report(void) __attribute__((destructor));
static void report(void) {
  fprintf(stderr,"SOCKET_SHIM send_calls=%llu recv_calls=%llu send_bytes=%llu recv_bytes=%llu delayed_calls=%llu delay_us=%lu min_bytes=%zu\n",
    atomic_load_explicit(&send_calls,memory_order_relaxed),
    atomic_load_explicit(&recv_calls,memory_order_relaxed),
    atomic_load_explicit(&send_bytes,memory_order_relaxed),
    atomic_load_explicit(&recv_bytes,memory_order_relaxed),
    atomic_load_explicit(&delayed_calls,memory_order_relaxed),delay_us,min_bytes);
}
