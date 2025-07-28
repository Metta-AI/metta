# Task: Debug Distributed Training Hang Issue

## Problem Statement
The distributed training system hangs when using multiple GPUs. The training process freezes and doesn't proceed beyond initialization. This needs to be debugged to identify the root cause and implement a fix.

## MVP Approach
Focus on identifying where the distributed training hangs by adding strategic logging and checking for common distributed training pitfalls like mismatched barrier calls, device placement issues, or incorrect process group initialization.

## Implementation Plan

### Phase 1: Information Gathering and Logging
1. **Add comprehensive logging to distributed initialization**
   - Log before and after `torch.distributed.init_process_group()`
   - Log device assignments for each rank
   - Log when each rank reaches key synchronization points

2. **Identify all barrier synchronization points**
   - Search for all `torch.distributed.barrier()` calls
   - Document where they occur in the training flow
   - Check for conditional barriers that might cause deadlock

3. **Check environment variables**
   - Log all distributed-related environment variables (RANK, LOCAL_RANK, WORLD_SIZE, etc.)
   - Verify they are set correctly for each process

### Phase 2: Common Issue Analysis
1. **Device placement verification**
   - Ensure models are moved to correct devices before DDP wrapping
   - Check that all tensors are on the same device within each rank
   - Verify device_ids and output_device in DDP initialization

2. **Model initialization consistency**
   - Ensure all ranks load the same initial weights
   - Check for any rank-specific model modifications before DDP
   - Verify BatchNorm → SyncBatchNorm conversion is happening correctly

3. **Data loading and seeding**
   - Check if data loading is properly distributed
   - Verify that random seeds are set differently per rank
   - Look for any I/O operations that might be blocking

### Phase 3: Systematic Debugging
1. **Create minimal reproduction case**
   - Strip down to minimal distributed setup
   - Test with dummy model and data
   - Gradually add components back to isolate issue

2. **Add timeout mechanisms**
   - Implement timeouts around potential hang points
   - Use `torch.distributed.monitored_barrier()` with timeout
   - Log which operations are timing out

3. **Check for deadlock patterns**
   - Look for mismatched collective operations
   - Verify all ranks participate in same collective calls
   - Check for conditional logic that might skip barriers

### Phase 4: Specific Areas to Investigate
1. **DistributedMettaAgent initialization**
   - The SyncBatchNorm conversion might be problematic
   - DDP device_ids configuration
   - Module attribute access through `__getattr__`

2. **Policy loading and checkpointing**
   - The `load_or_initialize_policy` function with distributed coordination
   - Checkpoint loading across ranks
   - Policy synchronization after loading

3. **VecEnv and async operations**
   - The `vecenv.async_reset()` call with rank-specific seeds
   - Zero-copy mode interactions with distributed training
   - AsyncFactor settings in distributed mode

4. **Training loop interactions**
   - Experience buffer sharing across ranks
   - Gradient synchronization timing
   - Stats collection and logging on non-master ranks

## Success Criteria
- [ ] Distributed training starts successfully without hanging
- [ ] All GPUs are utilized during training
- [ ] Training proceeds through multiple epochs
- [ ] Checkpointing works correctly across all ranks
- [ ] Performance scales appropriately with number of GPUs
- [ ] Clear error messages for common misconfigurations

## Implementation Updates
[This section will be updated during implementation]

## Debugging Commands
```bash
# Run with explicit distributed settings
TORCH_DISTRIBUTED_DEBUG=DETAIL python train.py

# Run with NCCL debug info
NCCL_DEBUG=INFO python train.py

# Run with Python faulthandler for deadlock detection
python -X faulthandler train.py

# Monitor GPU utilization
watch -n 1 nvidia-smi

# Check for hanging processes
ps aux | grep python | grep train
```

## Notes
- Consider that the hang might be environment-specific (CUDA version, NCCL version, PyTorch version)
- Document any version dependencies discovered
- Create unit tests for distributed components once issue is resolved