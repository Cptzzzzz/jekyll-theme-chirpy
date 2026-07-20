#include <cstdio>
#include <cstdlib>

#include <nccl.h>

ncclResult_t ncclGetDtree(
    int nranks,
    int rank,
    int* tree0_up,
    int* tree0_down0,
    int* tree0_down1,
    int* tree0_parent_child_type,
    int* tree1_up,
    int* tree1_down0,
    int* tree1_down1,
    int* tree1_parent_child_type);

int main(int argc, char** argv) {
  const int max_nodes = argc > 1 ? std::atoi(argv[1]) : 16;
  if (max_nodes < 1) return 2;

  std::puts(
      "nodes,rank,t0_up,t0_down0,t0_down1,t0_child_type,"
      "t1_up,t1_down0,t1_down1,t1_child_type");
  for (int nodes = 1; nodes <= max_nodes; ++nodes) {
    for (int rank = 0; rank < nodes; ++rank) {
      int t0_up, t0_down0, t0_down1, t0_type = -1;
      int t1_up, t1_down0, t1_down1, t1_type = -1;
      const ncclResult_t result = ncclGetDtree(
          nodes,
          rank,
          &t0_up,
          &t0_down0,
          &t0_down1,
          &t0_type,
          &t1_up,
          &t1_down0,
          &t1_down1,
          &t1_type);
      if (result != ncclSuccess) return 3;
      std::printf(
          "%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n",
          nodes,
          rank,
          t0_up,
          t0_down0,
          t0_down1,
          t0_type,
          t1_up,
          t1_down0,
          t1_down1,
          t1_type);
    }
  }
  return 0;
}
