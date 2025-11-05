# Monitor Metrics

## Overview

System resource monitoring

**Total metrics in this section:** 18

## Subsections

### Monitor

**Count:** 18 metrics

**cpu_count:** (1 value)
- `monitor/monitor/cpu_count`

**cpu_count_logical:** (1 value)
- `monitor/monitor/cpu_count_logical`

**cpu_count_physical:** (1 value)
- `monitor/monitor/cpu_count_physical`

**cpu_percent:** (1 value)
- `monitor/monitor/cpu_percent`

**gpu0_memory_percent:** (1 value)
- `monitor/monitor/gpu0_memory_percent`

**gpu0_memory_used_mb:** (1 value)
- `monitor/monitor/gpu0_memory_used_mb`

**gpu0_utilization:** (1 value)
- `monitor/monitor/gpu0_utilization`

**gpu_count:** (1 value)
- `monitor/monitor/gpu_count`

**gpu_memory_percent_avg:** (1 value)
- `monitor/monitor/gpu_memory_percent_avg`

**gpu_memory_used_mb_total:** (1 value)
- `monitor/monitor/gpu_memory_used_mb_total`

**gpu_utilization_avg:** (1 value)
- `monitor/monitor/gpu_utilization_avg`
  - Average GPU utilization across all available GPUs. (Unit: percentage)
  - **Interpretation:** Low values suggest compute bottleneck elsewhere. Aim for >80% during training.


**memory_available_mb:** (1 value)
- `monitor/monitor/memory_available_mb`

**memory_percent:** (1 value)
- `monitor/monitor/memory_percent`
  - System RAM usage percentage. (Unit: percentage)
  - **Interpretation:** High values (>90%) risk OOM errors. Consider reducing batch size or environment count.


**memory_total_mb:** (1 value)
- `monitor/monitor/memory_total_mb`

**memory_used_mb:** (1 value)
- `monitor/monitor/memory_used_mb`

**process_cpu_percent:** (1 value)
- `monitor/monitor/process_cpu_percent`

**process_memory_mb:** (1 value)
- `monitor/monitor/process_memory_mb`

**process_threads:** (1 value)
- `monitor/monitor/process_threads`


