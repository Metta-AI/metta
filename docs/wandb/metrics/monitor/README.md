# Monitor Metrics

## Overview

System resource monitoring

**Total metrics in this section:** 18

## Subsections

### Monitor

**Count:** 18 metrics

**cpu_count:**
- `monitor/monitor/cpu_count`

**cpu_count_logical:**
- `monitor/monitor/cpu_count_logical`

**cpu_count_physical:**
- `monitor/monitor/cpu_count_physical`

**cpu_percent:**
- `monitor/monitor/cpu_percent`

**gpu0_memory_percent:**
- `monitor/monitor/gpu0_memory_percent`

**gpu0_memory_used_mb:**
- `monitor/monitor/gpu0_memory_used_mb`

**gpu0_utilization:**
- `monitor/monitor/gpu0_utilization`

**gpu_count:**
- `monitor/monitor/gpu_count`

**gpu_memory_percent_avg:**
- `monitor/monitor/gpu_memory_percent_avg`

**gpu_memory_used_mb_total:**
- `monitor/monitor/gpu_memory_used_mb_total`

**gpu_utilization_avg:**
- `monitor/monitor/gpu_utilization_avg`
  Average GPU utilization across all available GPUs. (Unit: percentage)
  **Interpretation:** Low values suggest compute bottleneck elsewhere. Aim for >80% during training.


**memory_available_mb:**
- `monitor/monitor/memory_available_mb`

**memory_percent:**
- `monitor/monitor/memory_percent`
  System RAM usage percentage. (Unit: percentage)
  **Interpretation:** High values (>90%) risk OOM errors. Consider reducing batch size or environment count.


**memory_total_mb:**
- `monitor/monitor/memory_total_mb`

**memory_used_mb:**
- `monitor/monitor/memory_used_mb`

**process_cpu_percent:**
- `monitor/monitor/process_cpu_percent`

**process_memory_mb:**
- `monitor/monitor/process_memory_mb`

**process_threads:**
- `monitor/monitor/process_threads`


