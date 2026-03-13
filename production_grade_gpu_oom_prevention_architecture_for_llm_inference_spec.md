# Production-Grade Memory Control Plane for GPU OOM Prevention in LLM Inference

## 1. Document Status

- **Status:** Draft
- **Target audience:** Inference platform engineers, systems engineers, reliability engineers, infra reviewers, capacity planners
- **Target systems:** vLLM, Triton-based serving, Ray Serve, and custom LLM inference stacks
- **Primary focus:** Production-grade GPU OOM prevention through memory-aware scheduling and tiered KV cache control

---

## 2. Executive Summary

### 2.1 Problem

Production LLM inference systems do not fail from model size alone. They fail because GPU memory is typically managed as a best-effort runtime side effect instead of a scheduled and policy-governed resource.

The dominant GPU OOM causes in real-world inference are multi-dimensional:

- KV cache growth with long context
- activation spikes during prefill and decode
- overlap between prefill and decode peaks
- allocator fragmentation
- idle-session GPU residency
- concurrency oversubscription
- memory leaks and reference retention
- precision drift and silent dtype expansion
- restore storms and bandwidth collapse
- rank-local imbalance in distributed deployments

A stable inference system must therefore separate **memory policy** from the decoder loop and treat GPU memory as **schedulable capacity** rather than opportunistic free space.

### 2.2 Proposal

This document proposes a **memory-aware inference control plane** that:

- manages KV cache across **GPU, CPU RAM, NVMe, and shared storage**
- enforces **reservation-based admission control**
- isolates **prefill** from **decode**
- explicitly decides **restore vs recompute**
- prevents idle-session GPU hoarding
- provides crash-safe session recovery
- enforces tenant isolation, residency, and auditability
- exposes cost and latency signals as first-class control inputs

### 2.3 Intended Outcomes

Under the supported operating envelope, this architecture aims to deliver:

- near-elimination of avoidable GPU OOMs
- higher safe concurrent session capacity
- more predictable TTFT under mixed workloads
- reduced GPU waste from idle KV residency
- improved recovery behavior during node failure and rolling deploys
- measurable cost reductions through memory-aware scheduling

### 2.4 Core Thesis

The system should be understood as a **memory control plane for inference**, not merely an optimized decoder loop.

---

## 3. Goals

### 3.1 Primary Goals

- Prevent GPU OOM during LLM inference for interactive and long-context workloads
- Treat GPU memory as a reserved and schedulable resource
- Keep only actively decoding state on GPU whenever feasible
- Support fast warm resume when restore is cheaper than recompute
- Preserve low and predictable TTFT under bursty and multi-tenant load
- Provide production-grade recovery, observability, and policy tuning

### 3.2 Secondary Goals

- Reduce GPU cost per active user/session
- Increase safe concurrency without sacrificing reliability
- Enable portable integration across multiple inference engines
- Support per-tenant SLA tracking and cost attribution

---

## 4. Non-Goals

This design does **not** attempt to solve the following:

- training-time memory optimization
- optimizer or gradient state management
- arbitrary cross-model KV reuse
- exact infinite-context recall
- zero-cost live migration mid-token without replay/recovery
- backend kernel rewrites as the primary optimization strategy
- general-purpose distributed shared memory abstraction
- unrestricted 128K+ context support with exact fidelity and no tradeoffs

For very large context regimes, the design assumes that **windowing, sharding, summarization, or context compaction** may become necessary.

---

## 5. Design Principles

- **Memory is a first-class resource.** GPU memory must be reserved, budgeted, and governed by policy.
- **No decode without reservation.** Work is not admitted unless memory guarantees exist.
- **GPU is for active decode, not passive storage.** Idle session state should move off GPU.
- **Tier transitions must be explicit.** Every evict, restore, and invalidate action must be observable and auditable.
- **Correctness beats cache hit rate.** Reuse is only allowed when compatibility and tenancy guarantees hold.
- **Latency predictability outranks raw throughput.** Stable TTFT is often more valuable than peak token throughput.
- **Recovery is part of the design.** Crash, eviction, and deploy recovery paths are first-class, not bolted on later.
- **Policy must be tunable at runtime.** Hardcoded thresholds are operational risk.
- **Cost matters.** Recompute, restore, and residency decisions should be economically measurable.

---

## 6. Supported Operating Envelope

### 6.1 Model and Context Envelope

This architecture is intended for:

- model sizes from roughly **7B to 70B**
- interactive, document, IDE/copilot, and batch inference patterns
- context lengths up to **64K** as a first-class operational target
- contexts beyond **64K** with stronger controls such as:
  - KV sharding
  - windowed attention
  - summarization/compaction
  - context virtualization

### 6.2 Deployment Envelope

Supported deployment patterns include:

- single-GPU replicas
- tensor-parallel deployments
- pipeline-parallel deployments
- multi-tenant SaaS and internal platform deployments
- rolling upgrades with session continuity requirements

### 6.3 SLO Envelope

Exact numbers are deployment-specific, but the design assumes explicit targets for:

- TTFT P50 and P95 by cache tier
- OOM rate
- session recovery time objective
- restore latency by tier
- tier transition success rate
- fragmentation ceiling
- maximum acceptable idle GPU KV cost

---

## 7. Design Invariants

The following invariants define correctness boundaries for the system:

1. **No request is admitted unless memory reservation succeeds.**
2. **No decode begins while required KV state is unresolved or mid-restore.**
3. **GPU-resident KV is not authoritative once a successful colder-tier commit is recorded.**
4. **Metadata must not advertise a colder tier as authoritative until durable persistence completes.**
5. **No cross-tenant cache read is allowed under any condition.**
6. **Shared cache reuse requires strict compatibility validation.**
7. **Every tier transition must emit metrics and an audit event.**
8. **Session cleanup must return memory usage close to baseline within a configurable threshold.**
9. **Policy changes must be rollback-safe and must not corrupt authoritative session state.**
10. **A failed offload must never drop the last valid session copy.**

These invariants are more important than any individual heuristic.

---

## 8. Failure Taxonomy

### 8.1 Primary Failure Classes

GPU OOM in inference systems usually emerges from one or more of the following:

- **KV cache pressure:** long contexts and many resumed sessions retain too much state on GPU
- **Activation spikes:** prefill and decode create transient peaks beyond steady-state memory
- **Prefill/decode overlap:** concurrent large prefills plus active decodes create pathological peaks
- **Fragmentation:** free memory exists but cannot be allocated contiguously
- **Oversubscription:** scheduler admits work using point estimates rather than reservations
- **Idle session hoarding:** inactive sessions consume GPU memory long after decode stops
- **Precision drift:** accidental float32 or unsupported mixed precision doubles memory usage
- **Lifecycle leaks:** references remain after expected teardown
- **Restore storms:** resumed sessions saturate PCIe/NVMe before they saturate compute
- **Rank skew:** one shard/rank becomes memory-constrained before others in distributed execution

### 8.2 Why Simpler Approaches Fail

Common mitigations such as “reduce batch size” or “add more GPUs” are often insufficient because they ignore:

- workload heterogeneity
- session idleness and resumption
- tiering economics
- allocator behavior
- transient spikes
- recovery semantics
- multi-tenant fairness

The root problem is not merely **how much memory exists**, but **how memory is governed**.

---

## 9. System Overview

### 9.1 Mental Model

The system is a memory-aware control plane layered beside the inference engine:

```text
GPU (HBM)  <->  CPU RAM  <->  NVMe SSD  <->  Shared Storage
  hot            warm          cold           reusable/cross-node
```

### 9.2 Architectural Positioning

The inference engine (vLLM, Triton, custom backend) owns kernel execution and model forward passes.

The **Memory & Cache Manager** owns:

- cache residency policy
- tier transitions
- memory accounting
- restore scheduling
- authoritative session state
- eviction strategy

The **Scheduler** owns:

- admission control
- queue placement
- prefill/decode isolation
- restore-vs-recompute decisions
- concurrency enforcement

### 9.3 Core Components

- **Inference engine**
- **Memory & Cache Manager**
- **Reservation-based admission controller**
- **Scheduler**
- **Tiered persistence/storage interfaces**
- **Metadata and recovery subsystem**
- **Metrics, tracing, and audit pipeline**
- **Runtime policy/configuration service**

---

## 10. Component Responsibilities and Contracts

### 10.1 Scheduler Contract

The scheduler:

- decides whether a request is admitted
- reserves memory before dispatch
- selects restore vs recompute
- isolates prefill from decode
- enforces concurrency caps
- never dispatches decode without a satisfied reservation

### 10.2 Cache Manager Contract

The cache manager:

- owns session residency state
- performs evict/restore/invalidate transitions
- persists authoritative tier metadata
- enforces tier compatibility and authorization
- returns explicit success/failure states

### 10.3 Storage Contract

The storage subsystem:

- provides durable writes for colder tiers
- supports checksums and corruption detection
- provides encrypted persistence where required
- distinguishes transient store failure from data-not-found
- supports cleanup of orphaned partial artifacts

### 10.4 Inference Engine Contract

The inference engine:

- consumes GPU-resident tensors
- does not own long-term memory policy
- validates pointer/dtype/device assumptions before kernel launch
- exposes hooks for scheduler and cache manager integration

---

## 11. Canonical Data Model

A production design needs a **canonical persisted session record** separate from runtime-only handles.

### 11.1 Session Metadata

```python
from dataclasses import dataclass
from typing import Literal, Optional

dataclass
class SessionMemoryMeta:
    session_id: str
    tenant_id: str
    model_id: str
    workload_type: Literal["chat", "doc", "ide", "batch"]

    seq_len: int
    num_layers: int
    num_heads: int
    head_dim: int

    kv_bytes: int
    activation_bytes_peak: int

    precision: Literal["fp16", "bf16", "fp8", "int8", "int4"]

    # Authoritative residency and lifecycle
    tier: Literal["gpu", "cpu", "nvme", "shared"]
    state: Literal[
        "GPU_ACTIVE",
        "GPU_IDLE",
        "EVICTING_TO_CPU",
        "CPU_READY",
        "EVICTING_TO_NVME",
        "NVME_READY",
        "RESTORING",
        "INVALID",
        "RECOVERING",
    ]

    # Persisted locators
    kv_nvme_path: Optional[str] = None
    shared_key: Optional[str] = None

    # Compatibility and safety
    tokenizer_version: str = ""
    prompt_contract_hash: str = ""
    tensor_schema_version: str = "v1"
    data_zone: str = ""

    # Integrity and crypto
    blob_checksum: str = ""
    encryption_key_id: str = ""

    # Versioning and timestamps
    version: int = 0
    last_access_ts: float = 0.0
    updated_ts: float = 0.0
```

### 11.2 Runtime Handle

Runtime state may include non-persisted pointers:

```python
dataclass
class SessionMemoryHandle:
    meta: SessionMemoryMeta
    kv_gpu: Optional[torch.Tensor]
    kv_cpu: Optional[torch.Tensor]
```

### 11.3 Why This Separation Matters

Persisted metadata must be sufficient for:

- recovery after node failure
- rolling upgrades
- restore validation
- tier reconciliation
- audit and debugging

Pointers such as `kv_gpu` and `kv_cpu` are runtime details and are not authoritative after process loss.

---

## 12. Session State Machine

### 12.1 States

A formal state machine improves correctness and recovery:

- `GPU_ACTIVE` — currently decoding on GPU
- `GPU_IDLE` — still on GPU but no active decode
- `EVICTING_TO_CPU` — GPU->CPU copy in progress
- `CPU_READY` — authoritative warm copy in CPU memory
- `EVICTING_TO_NVME` — CPU->NVMe persistence in progress
- `NVME_READY` — authoritative cold copy on local NVMe
- `RESTORING` — colder tier is being rehydrated toward GPU
- `RECOVERING` — session is being reconstructed after failure or migration
- `INVALID` — no authoritative usable copy exists

### 12.2 Transition Rules

Each transition must define:

- trigger
- preconditions
- copy semantics
- durability semantics
- metadata update point
- timeout/retry behavior
- observability side effects

### 12.3 Atomicity Rule

Tier transitions follow **copy-complete-then-commit** semantics:

1. create or transfer new copy
2. validate success/checksum
3. persist metadata version update
4. only then release previous authoritative tier

This prevents loss of the last valid copy during partial failure.

---

## 13. Memory Tiering Architecture

### 13.1 Tier Definitions

- **GPU:** active decode only
- **CPU:** recently idle sessions with high resume probability
- **NVMe:** colder sessions with lower immediacy
- **Shared:** reusable or replicated state where correctness and policy allow

### 13.2 Baseline Tier Selection Policy

```python
def select_kv_tier(meta: SessionMemoryMeta) -> str:
    idle = time.time() - meta.last_access_ts

    if meta.workload_type == "chat":
        if idle < 60:
            return "cpu"
        return "nvme"

    if meta.workload_type == "doc":
        return "shared"

    if meta.workload_type in ("ide", "batch"):
        return "nvme"

    return "cpu"
```

### 13.3 Policy Inputs

Tier selection should consider:

- workload type
- idle duration
- expected resume probability
- context length
- restore time estimate
- storage and interconnect pressure
- data residency policy
- tenant-specific SLA class

### 13.4 Residency Principle

The core residency policy is:

- **actively decoding** sessions stay on GPU
- **recently inactive** sessions move to CPU
- **cold** sessions move to NVMe
- **reusable prefixes** may be placed in shared cache if correctness allows

---

## 14. Tier Transition Protocols

### 14.1 GPU -> CPU Offload

```python
def offload_kv_gpu_to_cpu(h: SessionMemoryHandle):
    with torch.no_grad():
        h.meta.state = "EVICTING_TO_CPU"
        cpu_copy = h.kv_gpu.to("cpu", non_blocking=True)
    torch.cuda.synchronize()
    h.kv_cpu = cpu_copy
    h.meta.tier = "cpu"
    h.meta.state = "CPU_READY"
    h.meta.version += 1
    h.kv_gpu = None
```

### 14.2 CPU -> NVMe Offload

```python
def offload_kv_cpu_to_nvme(h: SessionMemoryHandle):
    path = f"/nvme/kvcache/{h.meta.tenant_id}/{h.meta.session_id}.pt"
    tmp_path = f"{path}.tmp"

    try:
        h.meta.state = "EVICTING_TO_NVME"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(h.kv_cpu, tmp_path)
        os.replace(tmp_path, path)

        h.meta.kv_nvme_path = path
        h.meta.tier = "nvme"
        h.meta.state = "NVME_READY"
        h.meta.version += 1

        h.kv_cpu = None
    except OSError as e:
        logger.error("NVMe offload failed; retaining CPU tier", exc_info=e)
        metrics.increment("kv_nvme_offload_failure")
```

### 14.3 Shared Cache Store

```python
KV_SHARED_CACHE_TTL_S = int(os.getenv("SHARED_CACHE_TTL_S", "3600"))

def store_shared_kv(h: SessionMemoryHandle):
    key = f"{h.meta.tenant_id}:{h.meta.model_id}:{h.meta.prompt_contract_hash}"
    try:
        redis.setex(key, KV_SHARED_CACHE_TTL_S, serialize(h.kv_cpu))
        h.meta.shared_key = key
        h.meta.tier = "shared"
        h.meta.version += 1
    except redis.RedisError as e:
        logger.error("Shared cache store failed; retaining CPU tier", exc_info=e)
        metrics.increment("kv_shared_cache_store_failure")
```

### 14.4 Contract

A failed colder-tier write must **not** invalidate the warmer authoritative copy.

---

## 15. Reservation-Based Admission Control

### 15.1 Why Reservation Beats Estimation Alone

Point estimates are insufficient for stable scheduling. Production admission control must reserve for:

- KV cache growth
- activation peak
- restore buffers
- allocator overhead
- fragmentation reserve
- uncertainty margin
- distributed communication workspace where applicable

### 15.2 Reservation Model

```python
def estimate_request_memory(req):
    return (
        kv_bytes(req)
        + activation_peak(req)
        + restore_buffer_bytes(req)
        + fragmentation_reserve_bytes()
        + scheduler_safety_margin_bytes()
    )
```

### 15.3 Admission Policy

```python
def admit_request(req):
    est_mem = estimate_request_memory(req)
    if est_mem > free_gpu_budget():
        reject_or_defer(req)
        return False
    reserve_gpu_budget(req.request_id, est_mem)
    return True
```

### 15.4 Principle

**No request is admitted without a memory guarantee.**

---

## 16. Prefill and Decode Isolation

### 16.1 Why Isolation Matters

Prefill and decode have different memory shapes:

- prefill creates large transient activation spikes
- decode is usually lower per token but persistent over time
- overlapping large prefills with active decodes is a common OOM trigger

### 16.2 Policy

```python
def run_prefill(req):
    throttle_large_prefills()
    chunked_prefill(req)
    offload_kv_after_prefill(req)
```

Large prefills should be:

- chunked
- serialized when necessary
- isolated in dedicated scheduling lanes
- budgeted separately from decode

### 16.3 Queue Separation

Recommended queues:

- short-context interactive queue
- long-context interactive queue
- prefill queue
- decode queue
- restore queue
- batch queue

---

## 17. Restore vs Recompute Decision Framework

### 17.1 Inputs

The decision should consider:

- sequence length
- workload type
- restore latency estimate
- recompute latency estimate
- PCIe/NVLink bandwidth
- current restore queue depth
- cache hit probability
- current GPU memory pressure

### 17.2 Baseline Policy

```python
def schedule_request(req):
    meta = estimate_session_meta(req)

    recompute_time = estimate_recompute_time(meta)
    restore_time = estimate_restore_time(meta)

    if meta.seq_len > 8192 and restore_time < 0.7 * recompute_time:
        strategy = "restore"
    elif meta.workload_type == "batch":
        strategy = "recompute"
    elif restore_time < recompute_time:
        strategy = "restore"
    else:
        strategy = "recompute"

    projected_mem = meta.kv_bytes + meta.activation_bytes_peak
    if projected_mem > free_gpu_memory():
        reject_or_queue(req)
        return

    if strategy == "restore":
        prefetch_kv(meta.session_id)
        wait_for_kv_ready(meta.session_id)
    else:
        invalidate_existing_cache(meta.session_id)

    dispatch_decode(req)
```

### 17.3 Overload Rule

If the restore pipeline becomes saturated, the scheduler should:

- cap concurrent restores
- prioritize interactive sessions
- switch some requests to recompute
- defer low-priority resumes

---

## 18. Restore and Prefetch Architecture

### 18.1 Principle

Restore happens **before decode**, never inline inside the decoder critical path.

```python
def prefetch_kv(h):
    if h.meta.tier == "cpu":
        async_cpu_to_gpu(h)
    elif h.meta.tier == "nvme":
        async_nvme_to_gpu(h)
    elif h.meta.tier == "shared":
        async_shared_to_gpu(h)
```

### 18.2 Requirements for Robustness

A production restore path should include:

- bounded concurrent restores
- per-tier latency budgeting
- cancellation of stale restores
- integrity validation
- fallback to recompute when restore is slower or unavailable

---

## 19. Decode Scheduling and Activation Spike Control

### 19.1 Peak-Aware Scheduling

```python
def can_schedule_decode(meta):
    projected = meta.kv_bytes + meta.activation_bytes_peak
    return projected < available_gpu_memory()
```

### 19.2 Decode Staggering

```python
def decode_scheduler(queue):
    for req in queue:
        if req.phase == "decode":
            wait_for_other_peak_decodes()
        run(req)
```

### 19.3 Objective

Avoid synchronization of memory peaks across large decodes and prefills.

---

## 20. Allocator Discipline and Fragmentation Control

### 20.1 Problem

GPU memory fragmentation can cause allocation failure even when nominal free memory appears sufficient.

### 20.2 Strategy

```python
preallocate_kv_pool(max_kv_bytes)
preallocate_activation_pool(max_activation_bytes)
```

Recommended principles:

- fixed-size or paged KV blocks
- separate activation and KV pools
- avoid mixed-size ad hoc allocations
- reduce allocator churn
- reserve fragmentation headroom

### 20.3 Instrumentation

Track:

- free vs allocatable memory
- fragmentation ratio
- pool occupancy
- large allocation failures
- eviction pressure before OOM

---

## 21. Precision and Quantization Policy

### 21.1 Allowed KV Precision Modes

```python
VALID_KV_PRECISIONS = {
    torch.float16,
    torch.bfloat16,
    torch.float8_e4m3fn,
    torch.int8,
}
```

### 21.2 Validation

```python
def validate_precision(tensor: torch.Tensor):
    if tensor.dtype not in VALID_KV_PRECISIONS:
        raise TypeError(
            f"Unsupported KV cache precision: {tensor.dtype}. "
            f"Expected one of {VALID_KV_PRECISIONS}"
        )
```

### 21.3 Mixed Precision Compatibility

```python
def validate_mixed_precision_pipeline(weight_dtype: torch.dtype, kv_dtype: torch.dtype):
    mixed_ok = {
        (torch.int8, torch.float16),
        (torch.int8, torch.bfloat16),
        (torch.float8_e4m3fn, torch.float16),
        (torch.float8_e4m3fn, torch.bfloat16),
    }
    if (weight_dtype, kv_dtype) not in mixed_ok:
        raise ValueError(
            f"Unsupported mixed-precision pipeline: weights={weight_dtype}, kv={kv_dtype}"
        )
```

### 21.4 Operational Guidance

- accidental float32 promotion can silently double memory use
- INT8 KV is often viable for chat/document workloads
- INT4 KV requires workload-specific quality validation

---

## 22. Batch and Concurrency Control

### 22.1 Context-Aware Admission

```python
def admit_request(req):
    est_mem = estimate_kv(req) + estimate_activation(req)
    if est_mem > free_gpu_budget():
        reject_or_defer(req)
```

### 22.2 Queue Segmentation

Maintain separate queues for:

- short-context requests
- long-context requests
- batch/offline workloads
- interactive latency-sensitive workloads

### 22.3 Goal

Prevent long-context or restore-heavy traffic from collapsing interactive QoS.

---

## 23. Distributed and Multi-GPU Guardrails

### 23.1 Sharding Validation

```python
def validate_sharding():
    if kv_heads_per_rank != attention_heads / world_size:
        raise ValueError(
            f"KV head shard mismatch: {kv_heads_per_rank} per rank, "
            f"expected {attention_heads / world_size}"
        )
```

### 23.2 Required Additional Controls

Distributed deployments also require:

- per-rank memory accounting
- rank skew detection
- synchronized restore barriers where necessary
- shard version consistency
- recovery semantics when one rank loses state

### 23.3 Principle

A request is only as safe as its most memory-constrained rank.

---

## 24. Lifecycle Leak Prevention

### 24.1 Cleanup Contract

```python
def cleanup_session(session_id):
    free_kv(session_id)
    if not no_references_remain(session_id):
        raise RuntimeError(
            f"Memory leak detected: references remain after cleanup for session {session_id}"
        )
```

### 24.2 Invariant

Memory must return close to baseline after session teardown, within a configurable fragmentation-aware threshold.

### 24.3 Leak Detection

```python
def detect_gpu_memory_leak(window_minutes: int = 30, threshold_mb_per_min: float = 50.0):
    samples = gpu_residency_timeseries.last_n_minutes(window_minutes)
    if len(samples) < 4:
        return
    n = len(samples)
    xs = list(range(n))
    x_mean = sum(xs) / n
    y_mean = sum(samples) / n
    slope_mb_per_sample = (
        sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, samples))
        / sum((x - x_mean) ** 2 for x in xs)
    )
    rate = slope_mb_per_sample
    if rate > threshold_mb_per_min:
        logger.error(
            f"Possible GPU memory leak: trend +{rate:.1f} MB/min over {window_minutes} min"
        )
        metrics.increment("gpu_memory_leak_detected")
        alert_oncall(f"GPU memory leak suspected: {rate:.1f} MB/min")
```

---

## 25. Security, Tenant Isolation, and Compliance

### 25.1 Tenant-Scoped Cache Keys

```python
def make_cache_key(meta: SessionMemoryMeta) -> str:
    return f"{meta.tenant_id}:{meta.session_id}"
```

### 25.2 Authorized Restore

```python
def restore_shared_kv(h: SessionMemoryHandle) -> bool:
    key = make_cache_key(h.meta)
    if not authorize_cache_access(h.meta.tenant_id, key):
        raise PermissionError(
            f"Tenant {h.meta.tenant_id} not authorized to access cache key {key}"
        )
    try:
        raw = redis.get(key)
    except redis.RedisError as e:
        logger.error("Redis restore failed", exc_info=e)
        metrics.increment("kv_shared_cache_restore_failure")
        return False
    if raw is None:
        return False
    h.kv_cpu = deserialize(raw)
    return True
```

### 25.3 Encryption at Rest and in Transit

```python
import os
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

KV_ENCRYPTION_KEY = bytes.fromhex(os.environ["KV_AES256_KEY_HEX"])

def _make_aad(tenant_id: str, session_id: str) -> bytes:
    return f"{tenant_id}:{session_id}".encode()

def encrypt_kv(data: bytes, tenant_id: str, session_id: str) -> bytes:
    aesgcm = AESGCM(KV_ENCRYPTION_KEY)
    nonce = os.urandom(12)
    aad = _make_aad(tenant_id, session_id)
    return nonce + aesgcm.encrypt(nonce, data, aad)

def decrypt_kv(data: bytes, tenant_id: str, session_id: str) -> bytes:
    aesgcm = AESGCM(KV_ENCRYPTION_KEY)
    aad = _make_aad(tenant_id, session_id)
    return aesgcm.decrypt(data[:12], data[12:], aad)
```

Requirements:

- NVMe/local persistence: AES-256-GCM or platform-equivalent encryption
- shared cache transport: TLS
- inter-node replication: mTLS
- keys managed through secrets manager/HSM
- envelope encryption and rotation recommended in production

### 25.4 Data Residency

```python
REGION_TIER_MAP = {
    "eu-west":  ["gpu", "cpu", "nvme"],
    "us-east":  ["gpu", "cpu", "nvme", "shared"],
    "apac":     ["gpu", "cpu", "nvme"],
}

def select_kv_tier_with_residency(meta: SessionMemoryMeta) -> str:
    allowed_tiers = REGION_TIER_MAP.get(meta.data_zone, ["gpu", "cpu", "nvme"])
    preferred = select_kv_tier(meta)
    if preferred not in allowed_tiers:
        for tier in ["cpu", "nvme", "gpu"]:
            if tier in allowed_tiers:
                return tier
    return preferred
```

### 25.5 RBAC and Audit Logging

```python
ALLOWED_OPERATIONS: dict[str, set[str]] = {
    "evict":   {"cache_manager_service"},
    "restore": {"cache_manager_service", "inference_worker"},
    "read":    {"cache_manager_service", "inference_worker", "audit_service"},
}

def check_cache_rbac(operation: str, actor: str, tenant_id: str):
    if actor not in ALLOWED_OPERATIONS.get(operation, set()):
        raise PermissionError(
            f"Actor '{actor}' is not authorized to perform '{operation}' "
            f"on cache for tenant '{tenant_id}'"
        )
```

```python
def emit_audit_event(operation: str, meta: SessionMemoryMeta, actor: str):
    event = {
        "timestamp": time.time(),
        "operation": operation,
        "session_id": meta.session_id,
        "tenant_id": meta.tenant_id,
        "tier_from": meta.tier,
        "actor": actor,
        "data_zone": meta.data_zone,
    }
    audit_logger.info(json.dumps(event))
```

---

## 26. High Availability and Session Recovery

### 26.1 Metadata Durability

Persist metadata on every authoritative tier transition:

```python
SESSION_META_TTL_S = int(os.getenv("SESSION_META_TTL_S", "86400"))

def persist_session_meta(meta: SessionMemoryMeta):
    key = f"session_meta:{meta.tenant_id}:{meta.session_id}"
    redis_cluster.setex(key, SESSION_META_TTL_S, json.dumps(asdict(meta)))
```

### 26.2 Recovery After Node Failure

```python
def recover_session_after_node_failure(
    session_id: str, tenant_id: str
) -> SessionMemoryHandle:
    key = f"session_meta:{tenant_id}:{session_id}"
    raw = redis_cluster.get(key)
    if raw is None:
        raise SessionNotFoundError(
            f"Session {session_id} metadata missing for tenant {tenant_id}"
        )

    meta = SessionMemoryMeta(**json.loads(raw))
    meta.state = "RECOVERING"

    h = SessionMemoryHandle(
        meta=meta,
        kv_gpu=None,
        kv_cpu=None,
    )

    target_node = cluster_router.assign_healthy_node(meta.tenant_id)
    prefetch_kv(h)
    return h
```

### 26.3 Warm-Tier Replication

```python
def replicate_cpu_kv(h: SessionMemoryHandle, secondary_node: str):
    if h.meta.workload_type not in ("chat", "ide"):
        return
    try:
        remote_store.put(
            node=secondary_node,
            key=make_cache_key(h.meta),
            data=serialize(h.kv_cpu),
        )
    except RemoteStoreError as e:
        logger.warning(
            "CPU KV replication failed, continuing without replica", exc_info=e
        )
        metrics.increment("kv_replication_failure")
```

### 26.4 Graceful Degradation and Circuit Breakers

```python
class CircuitBreaker:
    def __init__(self, threshold: float = 0.85):
        self.threshold = threshold

    def check(self, gpu_memory_fraction: float):
        if gpu_memory_fraction > self.threshold:
            activate_fallback_model()
            shed_low_priority_requests()
            metrics.increment("circuit_breaker_open")
```

### 26.5 Rolling Upgrades

```python
def drain_replica_for_upgrade(replica_id: str, timeout_s: int = 120):
    scheduler.mark_draining(replica_id)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        active = scheduler.active_sessions_on(replica_id)
        if not active:
            break
        for h in active:
            if h.meta.state in ("GPU_IDLE", "GPU_ACTIVE"):
                offload_kv_gpu_to_cpu(h)
                persist_session_meta(h.meta)
        time.sleep(2)

    remaining = scheduler.active_sessions_on(replica_id)
    if remaining:
        for h in remaining:
            offload_kv_gpu_to_cpu(h)
            persist_session_meta(h.meta)

    scheduler.mark_ready_for_upgrade(replica_id)
```

---

## 27. Observability, Tracing, and SLO Monitoring

### 27.1 Metrics

```python
ttft = Histogram("ttft_seconds", labelnames=["tier", "tenant_id", "model_id"])
kv_hit = Counter("kv_cache_hit", labelnames=["tier", "tenant_id"])
restore_latency = Histogram("kv_restore_latency", labelnames=["tier", "tenant_id"])
recompute_ratio = Gauge("kv_recompute_ratio")
gpu_residency = Gauge("gpu_kv_bytes", labelnames=["tenant_id"])
gpu_memory_fragmentation = Gauge("gpu_memory_fragmentation_ratio")
cuda_oom_total = Counter("cuda_oom_total", labelnames=["tenant_id"])
tier_transition_failures = Counter("kv_tier_transition_failures", labelnames=["tier", "operation"])
```

### 27.2 OpenTelemetry Tracing

```python
from opentelemetry import trace
from opentelemetry.trace import SpanKind

tracer = trace.get_tracer("kv_cache_manager")

def offload_kv_gpu_to_cpu_traced(h: SessionMemoryHandle):
    with tracer.start_as_current_span(
        "kv_offload_gpu_to_cpu",
        kind=SpanKind.INTERNAL,
        attributes={
            "session.id": h.meta.session_id,
            "tenant.id": h.meta.tenant_id,
            "kv.bytes": h.meta.kv_bytes,
            "tier.from": "gpu",
            "tier.to": "cpu",
        },
    ) as span:
        offload_kv_gpu_to_cpu(h)
        span.set_attribute("transfer.success", True)
```

### 27.3 Dashboard Requirements

Dashboards should include:

- TTFT P50/P95 by tier and tenant
- GPU memory vs throughput
- prefill OOM vs decode OOM
- cache hit rate by workload type
- recompute ratio
- fragmentation ratio
- restore queue depth
- idle GPU KV cost

### 27.4 Alerts

Alert on:

- OOM spikes
- rising restore failures
- rising recompute ratio
- leak suspicion
- excessive fragmentation
- idle GPU KV cost breach
- canary regression

---

## 28. Cost Model and FinOps

### 28.1 Cost per TTFT

```python
cost_per_ttft = (
    gpu_seconds_to_first_token * gpu_cost_per_second
  + cpu_seconds_prefetch * cpu_cost_per_second
  + io_bytes_transferred * io_cost_per_byte
)
```

### 28.2 Cost per Resumed Session

```python
cost_per_resume = (
    restore_gpu_seconds * gpu_cost_per_second
  + restore_cpu_seconds * cpu_cost_per_second
  + storage_io_bytes * storage_cost_per_byte
)
```

### 28.3 Cost per Concurrent Session

```python
cost_per_concurrent_session = \
    (gpu_kv_bytes / total_gpu_memory) * gpu_hourly_cost
```

### 28.4 Operational Use

These metrics should guide:

- idle eviction thresholds
- restore-vs-recompute thresholds
- shared cache viability
- fallback model policy
- capacity planning

### 28.5 Executive View

Key panels:

- monthly GPU spend by TTFT tier
- cost per resumed session
- idle KV residency cost
- recompute waste
- effective concurrent capacity
- cost per active user

---

## 29. Numerical Worked Example: 7B vs 13B at 32K Context

Assumptions:

- A100 80GB at $2.50/hour
- CPU RAM cost equivalent: $0.00003/sec
- NVMe IO: $0.00000002/MB
- PCIe bandwidth: ~25 GB/s effective

### 29.1 KV Size Approximation

```text
KV size ≈ 2 * layers * heads * head_dim * seq_len * bytes_per_elem
```

7B (32 layers, 32 heads, fp16):
- ~16 GB KV cache at 32K tokens

13B (40 layers, 40 heads, fp16):
- ~28 GB KV cache at 32K tokens

### 29.2 Cold Recompute Cost

7B:
- prefill compute ≈ 1.2 sec
- TTFT cost ≈ $0.00083

13B:
- prefill compute ≈ 2.1 sec
- TTFT cost ≈ $0.00146

### 29.3 CPU -> GPU Restore Cost

7B:
- transfer 16 GB / 25 GB/s ≈ 0.64 sec
- cost ≈ $0.00044 GPU + $0.00002 CPU

13B:
- transfer 28 GB / 25 GB/s ≈ 1.12 sec
- cost ≈ $0.00078 GPU + $0.00004 CPU

### 29.4 Insight

Restore is materially cheaper than recompute for long contexts, especially as context grows.

---

## 30. Hardware Tradeoffs: A100 vs H100 vs H200

| GPU | HBM | NVLink BW | Spot Price (est.) | 7B KV @ 32K fits? | Sessions vs A100 |
|---|---|---|---|---|---|
| A100 80GB | 80 GB | PCIe ~25 GB/s | $2.50/hr | Yes | 1× baseline |
| H100 80GB | 80 GB | NVLink 900 GB/s | $3.50/hr | Yes | 1.4× |
| H200 141GB | 141 GB | NVLink 900 GB/s | $4.50/hr (est.) | Yes, large headroom | 2.5× |

### Implication

Higher-bandwidth interconnects shift the restore-vs-recompute frontier strongly toward restore.

---

## 31. 64K+ Context Breakpoints

At 64K+ context, new constraints dominate:

- single-session KV can approach or exceed per-GPU limits
- restore latency becomes human-noticeable
- fragmentation impact intensifies
- one session can monopolize safe capacity

### Required Extensions

- KV sharding across devices
- paged KV cache
- windowed attention
- context virtualization
- summarization/compaction

### Policy

```python
if meta.seq_len > 65536:
    summarize_and_compact(meta)
```

---

## 32. Backend Integration Plan

### 32.1 vLLM Integration

Key hook points:

- `SequenceGroup`
- `BlockManager`
- `Scheduler`

```python
from vllm.engine.cache import KVCache

class TieredKVCache(KVCache):
    def evict(self, block):
        offload_kv_gpu_to_cpu(block)

    def restore(self, block):
        prefetch_kv(block)
```

### 32.2 Triton / Custom Kernels

Triton kernels remain unchanged; tiering is enforced at pointer and scheduling boundaries.

```python
if not kv_tensor.is_cuda:
    raise ValueError("KV tensor must be on a CUDA device before kernel launch")
if kv_tensor.dtype != torch.float16:
    raise TypeError(f"KV tensor dtype must be float16 for Triton kernel, got {kv_tensor.dtype}")
```

### 32.3 Ray Serve

```python
@serve.deployment
class InferenceReplica:
    def __init__(self):
        self.cache_mgr = KVCacheManager()

    async def __call__(self, request):
        admit_request(request)
        return await decode(request)
```

Autoscaling should be driven by **memory pressure and TTFT**, not only QPS.

---

## 33. Validation Strategy

### 33.1 Unit and Contract Tests

Validate:

- state transitions
- precision compatibility
- persistence rules
- recovery path behavior
- authorization and residency rules

### 33.2 Memory Regression Tests

```python
GPU_LEAK_THRESHOLD_MB = float(os.getenv("GPU_LEAK_THRESHOLD_MB", "1.0"))

def get_gpu_memory_mb() -> float:
    return torch.cuda.memory_allocated() / (1024 ** 2)

@pytest.fixture(autouse=True)
def assert_no_gpu_memory_leak():
    torch.cuda.synchronize()
    baseline = get_gpu_memory_mb()
    yield
    torch.cuda.synchronize()
    final = get_gpu_memory_mb()
    leaked = final - baseline
    if leaked >= GPU_LEAK_THRESHOLD_MB:
        raise AssertionError(
            f"GPU memory leak detected: {leaked:.2f} MB above baseline after test "
            f"(threshold: {GPU_LEAK_THRESHOLD_MB} MB)"
        )
```

### 33.3 Load Testing

```python
async def load_test(
    concurrency: int,
    duration_s: int = 300,
    context_buckets: list[int]   = [2048, 16384, 65536],
    context_weights: list[float] = [0.3,  0.5,   0.2],
):
    context_lens = random.choices(context_buckets, weights=context_weights, k=concurrency)
    tasks = [
        synthetic_session(client, c, think_time_s=random.uniform(5, 30))
        for c in context_lens
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    ttfts = sorted(r for r in results if isinstance(r, float))
    print(f"P50 TTFT: {ttfts[len(ttfts) // 2]:.3f}s")
    print(f"P95 TTFT: {ttfts[int(len(ttfts) * 0.95)]:.3f}s")
    print(f"OOM count: {metrics.get('cuda_oom_total')}")
```

### 33.4 Chaos Testing

Required fault scenarios:

- NVMe failure during offload
- Redis/shared cache crash
- PCIe bandwidth degradation
- sudden GPU memory spike
- GPU node failure mid-decode
- Redis/session metadata expiry
- restore storm / queue saturation

### 33.5 Trace Replay and Shadow Policy Evaluation

Before changing thresholds or scheduler policy, replay production traces offline and compare:

- OOM risk
- TTFT
- restore queue depth
- cost per session
- rejected request rate

This is essential for safe rollout.

---

## 34. Canary Rollout and Policy Tuning

### 34.1 Runtime Configuration

```python
@dataclass
class InferenceMemoryConfig:
    chat_cpu_idle_threshold_s: float = float(os.getenv("CHAT_CPU_IDLE_S", "60"))
    chat_nvme_idle_threshold_s: float = float(os.getenv("CHAT_NVME_IDLE_S", "300"))
    eviction_check_interval_s: float = float(os.getenv("EVICTION_INTERVAL_S", "10"))
    restore_ratio_threshold: float = float(os.getenv("RESTORE_RATIO", "0.7"))
    context_windowing_threshold: int = int(os.getenv("CONTEXT_WINDOW_THRESHOLD", "65536"))
    shared_cache_ttl_s: int = int(os.getenv("SHARED_CACHE_TTL_S", "3600"))
    session_meta_ttl_s: int = int(os.getenv("SESSION_META_TTL_S", "86400"))
    circuit_breaker_memory_threshold: float = float(os.getenv("CB_MEMORY_THRESHOLD", "0.85"))
    policy_variant: str = os.getenv("POLICY_VARIANT", "default")
```

### 34.2 Policy Variants

```python
POLICY_VARIANTS = {
    "default":            {"restore_ratio_threshold": 0.70, "chat_cpu_idle_threshold_s": 60},
    "aggressive_offload": {"restore_ratio_threshold": 0.60, "chat_cpu_idle_threshold_s": 30},
    "conservative":       {"restore_ratio_threshold": 0.80, "chat_cpu_idle_threshold_s": 120},
}
```

### 34.3 Canary Gate

```python
def canary_policy_gate(canary_metrics: dict, control_metrics: dict) -> bool:
    if canary_metrics["oom_rate"] > 2 * control_metrics["oom_rate"]:
        logger.error("Canary OOM rate exceeded threshold; rolling back policy")
        return False
    if canary_metrics["ttft_p95"] > 1.5 * control_metrics["ttft_p95"]:
        logger.error("Canary TTFT P95 degraded; rolling back policy")
        return False
    return True
```

### 34.4 Rollout Phases

1. metrics and shadow estimation only
2. admission control enabled
3. low-risk tiered offload enabled
4. restore-vs-recompute policy enabled on canary
5. full rollout with automated rollback gates

---

## 35. Failure Modes and Guard Summary

| Failure | Guard |
|---|---|
| KV hoarding | Idle eviction + tiering |
| Fragmentation | Fixed pools / paged blocks |
| Activation spikes | Peak-aware scheduling |
| Batch collapse | Context-aware batching |
| Prefill OOM | Chunked prefill |
| Precision drift | Runtime dtype validation |
| Parallelism duplication | Shard audits and per-rank accounting |
| Memory leaks | Lifecycle invariants and regression tests |
| Burst overload | Reservation-based admission control |
| Restore storm | Bounded restore concurrency + recompute fallback |

---

## 36. Alternatives Considered

### 36.1 GPU-Only Residency
Rejected because idle sessions accumulate and long-context concurrency collapses.

### 36.2 Recompute-Only Strategy
Simpler but often more expensive and slower for resumed long-context sessions.

### 36.3 CPU-Only Warm Tier Without NVMe
Useful for short idleness but insufficient for larger fleets and colder sessions.

### 36.4 Remote Object Store Spill
Operationally simple but generally too slow for low-latency warm resumes.

### 36.5 Summarization-First Design
Powerful for very long contexts but not a replacement for base memory policy.

---

## 37. Risks and Open Questions

- shared cache invalidation complexity under model/tokenizer changes
- quality degradation envelope for INT4 KV
- restore storms during synchronized user return
- heterogeneous hardware fleets and threshold portability
- per-rank skew in large tensor-parallel deployments
- break-even points for summarization vs restore at very large contexts
- storage format choice for production NVMe tier beyond simple serialized tensors

---

## 38. Executive Synthesis

This design reframes LLM inference as a **memory-governed distributed systems problem** rather than a decoder-only optimization problem.

If implemented with the invariants and contracts in this document:

- engineers gain deterministic memory behavior
- operators gain recovery and observability
- finance gains measurable unit economics
- leadership gains predictable scaling characteristics

The key strategic idea is simple:

> **Treat KV cache as state, not compute.**

Once session memory is governed as tiered, schedulable, auditable state, GPU OOM prevention becomes an engineering discipline rather than an emergency reaction.

---

## 39. Appendix A: Decision Rule of Thumb

```text
If restore_time < 0.7 * recompute_time -> restore
Else -> recompute
```

This threshold must be treated as runtime policy, not code constant.

---

## 40. Appendix B: 64K+ Economic Reality

At very large context windows, the marginal cost per token rises superlinearly. At that point, exact replay of full context often becomes less economical than:

- summarization
- context compaction
- hot-span restoration
- sharded or paged KV strategies

---

## 41. Appendix C: One-Line Outcome Statement

**We reduce GPU spend and OOM risk by treating inference memory as governed state rather than accidental runtime residue.