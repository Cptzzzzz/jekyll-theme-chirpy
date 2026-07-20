# Distributed Fault Matrix

Run ID: `20260710T015721Z`

| case | distributed debug | return code | external timeout | classified first error |
|---|---|---:|---:|---|
| skip | OFF | 1 | 0 | Watchdog caught collective operation timeout: WorkNCCL(SeqNum=3, OpType=ALLREDUCE, NumelIn=1, NumelOut=1, Timeout(ms)=6000) ran for 6007 milliseconds before timing out. |
| op_mismatch | DETAIL | 1 | 0 | Detected mismatch between collectives on ranks. Rank 0 is running collective: CollectiveFingerPrint(SequenceNumber=2, OpType=BROADCAST, TensorShape=[1], TensorDtypes=Float, TensorDeviceTypes=TensorOptions(dtype=float (default), device=cuda, layout=Strided (default), requires_grad=false (default), pinned_memory=false (default), memory_format=(nullopt))), but Rank 1 is running collective: CollectiveFingerPrint(SequenceNumber=2, OpType=ALLREDUCE, TensorShape=[1], TensorDtypes=Float, TensorDeviceTyp |
| size_mismatch | DETAIL | 1 | 0 | Detected mismatch between collectives on ranks. Rank 1 is running collective: CollectiveFingerPrint(SequenceNumber=2, OpType=ALLREDUCE, TensorShape=[8], TensorDtypes=Float, TensorDeviceTypes=TensorOptions(dtype=float (default), device=cuda, layout=Strided (default), requires_grad=false (default), pinned_memory=false (default), memory_format=(nullopt))), but Rank 0 is running collective: CollectiveFingerPrint(SequenceNumber=2, OpType=ALLREDUCE, TensorShape=[4], TensorDtypes=Float, TensorDeviceTyp |
| rank_exit | OFF | 1 | 0 | ChildFailedError: |

All cases first completed a valid AllReduce, so the communicator and P2P path were healthy before fault injection. The failure signature therefore belongs to collective ordering or peer liveness, not initial network setup.
