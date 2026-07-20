#include "nccl_tuner.h"

#include <cstddef>

namespace {

constexpr std::size_t kTargetBytes = 64ull << 20;

ncclResult_t tunerInit(
    std::size_t /*nRanks*/,
    std::size_t /*nNodes*/,
    ncclDebugLogger_t /*logFunction*/,
    void** context) {
  *context = nullptr;
  return ncclSuccess;
}

ncclResult_t tunerGetCollInfo(
    void* /*context*/,
    ncclFunc_t collType,
    std::size_t nBytes,
    int /*numPipeOps*/,
    float** collCostTable,
    int numAlgo,
    int numProto,
    int* nChannels) {
  if (collType != ncclFuncAllReduce || nBytes != kTargetBytes) {
    return ncclSuccess;
  }

  // NCCL passes a contiguous [algorithm][protocol] table through a float** ABI.
  // Keep unsupported entries at -1 and make every supported alternative costly.
  auto* costs = reinterpret_cast<float*>(collCostTable);
  for (int algorithm = 0; algorithm < numAlgo; ++algorithm) {
    for (int protocol = 0; protocol < numProto; ++protocol) {
      float& cost = costs[algorithm * numProto + protocol];
      if (cost != NCCL_ALGO_PROTO_IGNORE) cost = 1.0e9f;
    }
  }

  // A zero cost wins NCCL's subsequent minimum search. Two channels should
  // therefore appear as gridX=2 in the selected collective kernel launch.
  costs[NCCL_ALGO_TREE * numProto + NCCL_PROTO_SIMPLE] = 0.0f;
  *nChannels = 2;
  return ncclSuccess;
}

ncclResult_t tunerDestroy(void* /*context*/) {
  return ncclSuccess;
}

}  // namespace

extern "C" {
ncclTuner_v3_t ncclTunerPlugin_v3 = {
    "ch16-force-tree-simple",
    tunerInit,
    tunerGetCollInfo,
    tunerDestroy,
};
}
