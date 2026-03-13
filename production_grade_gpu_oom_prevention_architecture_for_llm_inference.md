# Production-Grade GPU OOM Prevention Architecture for LLM Inference

This document consolidates *all major GPU OOM dimensions* encountered in real-world LLM inference systems and provides a unified, production-realistic implementation blueprint. It extends beyond KV-cache offloading to cover allocator behavior, scheduling, batching, prefill imbalance, parallelism artifacts, lifecycle leaks, precision drift, and admission control. The goal is not to optimize a model, but to engineer a resilient inference platform.

---

## I. System Architecture (Mental Model)

You are building a **memory-aware inference control plane**, not a decoder loop.

```
GPU (HBM)  ←→  CPU RAM  ←→  NVMe SSD  ←→  Shared Storage
  ↑             ↑            ↑               ↑
 Active      Warm idle     Cold idle     Cross-user reuse
```

The inference engine (vLLM / Triton / custom) never owns memory policy. A **Memory & Cache Manager** does.

Core principles:
- GPU holds only *currently decoding* state
- CPU RAM holds *recently inactive* session state
- NVMe / shared storage holds *cold or reusable* state
- Prefill, decode, and eviction are first-class phases
- TTFT and memory predictability dominate throughput

---

## II. Unified Data Model

### Session Memory Metadata

```python
@dataclass
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
    tier: Literal["gpu", "cpu", "nvme", "shared"]

    last_access_ts: float
    phase: Literal["prefill", "decode", "idle"]
    data_zone: str = ""   # geo/regulatory residency constraint (GDPR, HIPAA); must be set explicitly at session creation
```

### Memory Handles

```python
@dataclass
class SessionMemoryHandle:
    meta: SessionMemoryMeta
    kv_gpu: Optional[torch.Tensor]
    kv_cpu: Optional[torch.Tensor]
    kv_nvme_path: Optional[str]
    shared_key: Optional[str]
```

---

## III. KV Cache Tiered Offloading (Primary Dimension)

### Tier Selection Policy

```python
def select_kv_tier(meta: SessionMemoryMeta) -> str:
    idle = time.time() - meta.last_access_ts

    if meta.workload_type == "chat":
        return "cpu" if idle < 60 else "nvme"

    if meta.workload_type == "doc":
        return "shared"

    if meta.workload_type in ("ide", "batch"):
        return "nvme"

    return "cpu"
```

### GPU → CPU Offload

```python
def offload_kv_gpu_to_cpu(h: SessionMemoryHandle):
    with torch.no_grad():
        h.kv_cpu = h.kv_gpu.to("cpu", non_blocking=True)
    torch.cuda.synchronize()   # ensure transfer completes before releasing GPU tensor
    h.kv_gpu = None
    h.meta.tier = "cpu"
```

### CPU → NVMe Offload

```python
def offload_kv_cpu_to_nvme(h: SessionMemoryHandle):
    path = f"/nvme/kvcache/{h.meta.tenant_id}/{h.meta.session_id}.pt"
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(h.kv_cpu, path)
        h.kv_cpu = None
        h.kv_nvme_path = path
        h.meta.tier = "nvme"
    except OSError as e:
        logger.error("NVMe offload failed, retaining CPU tier", exc_info=e)
        metrics.increment("kv_nvme_offload_failure")
        # fallback: keep data in CPU tier; do not clear h.kv_cpu
```

### Shared Cache (Document Reuse)

```python
KV_SHARED_CACHE_TTL_S = int(os.getenv("SHARED_CACHE_TTL_S", "3600"))

def store_shared_kv(h: SessionMemoryHandle):
    key = f"{h.meta.tenant_id}:{hash_prefix(h.meta)}"
    try:
        redis.setex(key, KV_SHARED_CACHE_TTL_S, serialize(h.kv_cpu))
        h.shared_key = key
        h.meta.tier = "shared"
    except redis.RedisError as e:
        logger.error("Redis shared cache store failed, retaining CPU tier", exc_info=e)
        metrics.increment("kv_shared_cache_store_failure")
        # fallback: keep data in CPU tier; do not update tier
```

---

## IV. Async Prefetch and Fast Restore

```python
def prefetch_kv(h):
    if h.meta.tier == "cpu":
        async_cpu_to_gpu(h)
    elif h.meta.tier == "nvme":
        async_nvme_to_gpu(h)
    elif h.meta.tier == "shared":
        async_shared_to_gpu(h)
```

Restore always happens **before decode**, never inline.

---

## V. Activation Memory Spike Control

### Peak-Aware Decode Scheduling

```python
def can_schedule_decode(meta):
    projected = meta.kv_bytes + meta.activation_bytes_peak
    return projected < available_gpu_memory()
```

### Decode Staggering

```python
def decode_scheduler(queue):
    for req in queue:
        if req.phase == "decode":
            wait_for_other_peak_decodes()
        run(req)
```

---

## VI. Prefill vs Decode Isolation

```python
def run_prefill(req):
    throttle_large_prefills()
    chunked_prefill(req)
    offload_kv_after_prefill(req)
```

Large prefills are serialized or chunked to avoid transient OOM spikes.

---

## VII. Batch and Concurrency Control

### Context-Aware Admission Control

```python
def admit_request(req):
    est_mem = estimate_kv(req) + estimate_activation(req)
    if est_mem > free_gpu_budget():
        reject_or_defer(req)
```

### Separate Queues

- short-context batch queue
- long-context batch queue

---

## VIII. Fragmentation and Allocator Discipline

```python
# Startup
preallocate_kv_pool(max_kv_bytes)
preallocate_activation_pool(max_activation_bytes)
```

Avoid mixed-size allocations and allocator churn.

---

## IX. Model Parallelism Guardrails

```python
def validate_sharding():
    if kv_heads_per_rank != attention_heads / world_size:
        raise ValueError(
            f"KV head shard mismatch: {kv_heads_per_rank} per rank, "
            f"expected {attention_heads / world_size}"
        )
```

Audit per-rank memory usage continuously.

---

## X. Lifecycle Leak Prevention

```python
def cleanup_session(session_id):
    free_kv(session_id)
    if not no_references_remain(session_id):
        raise RuntimeError(
            f"Memory leak detected: references remain after cleanup for session {session_id}"
        )
```

Add invariants: memory must return to baseline.

---

## XI. Precision Enforcement

```python
VALID_KV_PRECISIONS = {
    torch.float16,
    torch.bfloat16,
    torch.float8_e4m3fn,   # fp8 (Hopper / Ada Lovelace)
    torch.int8,             # INT8 quantized KV (KV-specific quantization, distinct from GPTQ/AWQ weight quantization)
}

def validate_precision(tensor: torch.Tensor):
    if tensor.dtype not in VALID_KV_PRECISIONS:
        raise TypeError(
            f"Unsupported KV cache precision: {tensor.dtype}. "
            f"Expected one of {VALID_KV_PRECISIONS}"
        )
```

KV cache precision drift silently causes OOMs — a stray `float32` upcast doubles memory consumption.

### INT8 / INT4 Quantized KV Cache

Production deployments increasingly use quantized KV caches to reduce memory footprint:

```python
def validate_mixed_precision_pipeline(weight_dtype: torch.dtype, kv_dtype: torch.dtype):
    """Validate that weight quantization and KV cache precision are compatible."""
    mixed_ok = {
        (torch.int8, torch.float16),          # INT8 weights (GPTQ/AWQ), fp16 KV cache
        (torch.int8, torch.bfloat16),
        (torch.float8_e4m3fn, torch.float16), # fp8 weights, fp16 KV (Hopper)
        (torch.float8_e4m3fn, torch.bfloat16),
    }
    if (weight_dtype, kv_dtype) not in mixed_ok:
        raise ValueError(
            f"Unsupported mixed-precision pipeline: weights={weight_dtype}, kv={kv_dtype}"
        )
```

Memory savings at 32K context (7B model):

| Precision | KV Cache Size | Quality Impact |
|---|---|---|
| fp16 | ~16 GB | Baseline |
| int8 | ~8 GB | <1–2% perplexity delta |
| int4 | ~4 GB | 2–5% perplexity delta — evaluate per workload |

INT8 KV caches are viable for most chat and document workloads. INT4 requires per-deployment quality validation.

---

## XII. Control Plane Memory Prediction

```python
def estimate_request_memory(req):
    return kv_bytes(req) + activation_peak(req)
```

GPU memory is schedulable capacity, not best-effort.

---

## XIII. Monitoring and KPIs

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

All metrics carry a `tenant_id` label for per-customer SLA tracking and cost allocation.

Dashboards:
- TTFT P50 / P95 by tier and tenant
- GPU memory vs throughput
- Prefill OOMs vs decode OOMs
- Cache hit rate by workload
- Recompute ratio
- GPU memory fragmentation ratio

### OpenTelemetry Distributed Tracing

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

Wrap all tier transitions (evict, restore, prefetch) in spans. Correlate TTFT with tier transition latency in your distributed tracing backend (Jaeger, Tempo, or AWS X-Ray).

### Anomaly Detection: Slow Memory Leak Detection

```python
def detect_gpu_memory_leak(window_minutes: int = 30, threshold_mb_per_min: float = 50.0):
    """Flag if GPU KV residency has a sustained upward trend over `window_minutes`.

    Uses a linear regression slope rather than strict monotonic detection, tolerating
    short-term noise and small dips while still catching genuine slow leaks.
    """
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
    # Convert sample-based slope to MB/min assuming one sample per minute
    rate = slope_mb_per_sample
    if rate > threshold_mb_per_min:
        logger.error(
            f"Possible GPU memory leak: trend +{rate:.1f} MB/min over {window_minutes} min"
        )
        metrics.increment("gpu_memory_leak_detected")
        alert_oncall(f"GPU memory leak suspected: {rate:.1f} MB/min")
```

---

## XIV. Failure Modes and Guards

| Failure | Guard |
|------|------|
| KV hoarding | Idle eviction + TTL |
| Fragmentation | Fixed pools |
| Activation spikes | Staggered decode |
| Batch collapse | Context-aware batching |
| Prefill OOM | Chunked prefill |
| Parallelism duplication | Shard audits |
| Memory leaks | Lifecycle invariants |
| Precision drift | Runtime validation |
| Burst overload | Admission control |

---

## XV. Cost Models and Economic Signals

Modern inference platforms fail as often from cost blindness as from OOMs. Once KV cache and memory are treated as state, cost becomes quantifiable and optimizable.

### A. Cost per TTFT

TTFT cost captures the *economic penalty of latency*.

```python
cost_per_ttft = (
    gpu_seconds_to_first_token * gpu_cost_per_second
  + cpu_seconds_prefetch * cpu_cost_per_second
  + io_bytes_transferred * io_cost_per_byte
)
```

Key drivers:
- Cold recompute TTFT is dominated by GPU compute cost
- Warm resume TTFT shifts cost to IO and memory transfer
- Optimal systems minimize GPU seconds before first token

This metric allows direct comparison of:
- recompute vs cache restore
- GPU vs CPU-resident cache
- SSD vs network-backed restore

### B. Cost per Resumed Session

```python
cost_per_resume = (
    restore_gpu_seconds * gpu_cost_per_second
  + restore_cpu_seconds * cpu_cost_per_second
  + storage_io_bytes * storage_cost_per_byte
)
```

This metric reveals when offloading tiers stop being economical. If restore cost exceeds recompute cost, the tier is misclassified.

### C. Cost per Concurrent Session

```python
cost_per_concurrent_session = 
    (gpu_kv_bytes / total_gpu_memory) * gpu_hourly_cost
```

Idle KV cache inflates this cost silently. Tiered eviction is the primary lever.

---

## XVI. Mapping to vLLM, Triton, and Ray Serve

This architecture maps cleanly onto modern inference stacks with minimal invasive changes.

### A. vLLM Integration

vLLM already exposes KV cache as a managed object.

Key integration points:
- Hook into `KVCacheManager` in the scheduler
- Override default GPU-only cache residency
- Use vLLM block manager to segment KV cache by session

```python
from vllm.engine.cache import KVCache

class TieredKVCache(KVCache):
    def evict(self, block):
        offload_kv_gpu_to_cpu(block)

    def restore(self, block):
        prefetch_kv(block)
```

Use vLLM’s sequence groups to classify workload type and idle time.

### B. Triton / Custom Kernel Integration

Triton kernels remain unchanged. The integration happens at the *pointer level*.

- KV tensors passed to Triton are restored GPU pointers
- Offloading logic lives outside kernel execution
- Precision enforcement is validated before kernel launch

```python
if not kv_tensor.is_cuda:
    raise ValueError("KV tensor must be on a CUDA device before kernel launch")
if kv_tensor.dtype != torch.float16:
    raise TypeError(f"KV tensor dtype must be float16 for Triton kernel, got {kv_tensor.dtype}")
```

This preserves kernel performance while enabling tiered memory.

### C. Ray Serve Integration

Ray Serve provides the control plane primitives.

Mapping:
- One Ray actor per GPU = memory isolation domain
- KV cache manager runs as a colocated actor
- Shared cache backed by Ray object store or Redis

```python
@serve.deployment
class InferenceReplica:
    def __init__(self):
        self.cache_mgr = KVCacheManager()

    async def __call__(self, request):
        admit_request(request)
        return await decode(request)
```

Ray’s autoscaling should be driven by **memory pressure and TTFT**, not QPS.

---

## XVII. Tenant Isolation & Data Security

In multi-tenant deployments (SaaS, internal platforms), KV cache tiers hold encoded representations of user inputs. Isolation and confidentiality are non-negotiable.

### A. Namespaced Cache Keys

All cache keys — CPU heap references, NVMe paths, Redis keys — must be scoped to tenant:

```python
def make_cache_key(meta: SessionMemoryMeta) -> str:
    return f"{meta.tenant_id}:{meta.session_id}"

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

### B. Encryption at Rest and in Transit

KV cache on NVMe and shared storage contains implicit semantic context of user conversations and must be encrypted.

```python
import os
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

KV_ENCRYPTION_KEY = bytes.fromhex(os.environ["KV_AES256_KEY_HEX"])  # 32-byte key from secrets manager

def _make_aad(tenant_id: str, session_id: str) -> bytes:
    """Bind ciphertext to its owner — prevents cross-tenant ciphertext substitution attacks."""
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

def offload_kv_cpu_to_nvme_encrypted(h: SessionMemoryHandle):
    path = f"/nvme/kvcache/{h.meta.tenant_id}/{h.meta.session_id}.pt"
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        raw = serialize(h.kv_cpu)
        with open(path, "wb") as f:
            f.write(encrypt_kv(raw, h.meta.tenant_id, h.meta.session_id))
        h.kv_cpu = None
        h.kv_nvme_path = path
        h.meta.tier = "nvme"
    except OSError as e:
        logger.error("Encrypted NVMe offload failed, retaining CPU tier", exc_info=e)
        metrics.increment("kv_nvme_offload_failure")
```

Encryption requirements:
- NVMe storage: AES-256-GCM at rest (key from HSM / secrets manager)
- Redis shared cache: TLS transport (`rediss://`) with rotating AUTH tokens
- Inter-node KV cache replication: mutual TLS (mTLS)

### C. Data Residency / Sovereignty

```python
REGION_TIER_MAP = {
    "eu-west":  ["gpu", "cpu", "nvme"],   # no cross-region shared cache (GDPR)
    "us-east":  ["gpu", "cpu", "nvme", "shared"],
    "apac":     ["gpu", "cpu", "nvme"],   # configurable per regulatory requirement
}

def select_kv_tier_with_residency(meta: SessionMemoryMeta) -> str:
    allowed_tiers = REGION_TIER_MAP.get(meta.data_zone, ["gpu", "cpu", "nvme"])
    preferred = select_kv_tier(meta)
    if preferred not in allowed_tiers:
        # Fall back to the warmest allowed tier to minimize TTFT impact
        for tier in ["cpu", "nvme", "gpu"]:
            if tier in allowed_tiers:
                return tier
    return preferred
```

GDPR/HIPAA workloads: `data_zone` is enforced at session admission; the shared Redis tier is disabled for EU and APAC regions unless data residency is explicitly confirmed.

### D. RBAC on Cache Operations

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

### E. Audit Logging

Every tier transition emits a structured audit event:

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
    audit_metrics.increment(
        "cache_audit_event",
        tags={"operation": operation, "tenant": meta.tenant_id},
    )
```

Audit events feed into a SIEM (e.g., Splunk, Datadog Logs) for compliance reporting and anomaly detection.

---

## XVIII. Numerical Worked Example: 7B vs 13B at 32K Context

Assumptions (illustrative but realistic):
- GPU: A100 80GB at $2.50/hour ($0.000694/sec)
- CPU RAM: $0.00003/sec equivalent
- NVMe IO: $0.00000002/MB
- PCIe bandwidth: ~25 GB/s effective

### KV Cache Size

Approximate KV cache size:

```text
KV size ≈ 2 * layers * heads * head_dim * seq_len * bytes_per_elem
```

7B (32 layers, 32 heads, fp16):
- ~16 GB KV cache at 32K tokens

13B (40 layers, 40 heads, fp16):
- ~28 GB KV cache at 32K tokens

### Cold Recompute Cost

7B recompute:
- Prefill compute ≈ 1.2 sec
- TTFT cost ≈ $0.00083

13B recompute:
- Prefill compute ≈ 2.1 sec
- TTFT cost ≈ $0.00146

### Cache Restore Cost (CPU → GPU)

7B restore:
- Transfer 16 GB / 25 GB/s ≈ 0.64 sec
- Cost ≈ $0.00044 (GPU) + $0.00002 (CPU)

13B restore:
- Transfer 28 GB / 25 GB/s ≈ 1.12 sec
- Cost ≈ $0.00078 (GPU) + $0.00004 (CPU)

### Insight

- Restore is ~1.8× cheaper than recompute at 7B
- Restore is ~1.9× cheaper than recompute at 13B
- Cost gap widens further with longer context

### H100 / H200 Hardware Comparison

| GPU | HBM | NVLink BW | Spot Price (est.) | 7B KV @ 32K fits? | Sessions vs A100 |
|---|---|---|---|---|---|
| A100 80GB | 80 GB | PCIe ~25 GB/s | $2.50/hr | Yes (16 GB) | 1× baseline |
| H100 80GB | 80 GB | NVLink 900 GB/s | $3.50/hr | Yes (16 GB) | 1.4× (faster restore) |
| H200 141GB | 141 GB | NVLink 900 GB/s | $4.50/hr (est.) | Yes, with large headroom | 2.5× (more HBM) |

With H200 141 GB HBM, sessions that previously required NVMe offload at 32K context can remain fully GPU-resident, eliminating restore latency for those tiers. Tier transition thresholds should be recalibrated per hardware generation.

**CXL Memory Pooling**: An emerging tier between CPU RAM and NVMe. CXL 2.0/3.0 attached memory pools (e.g., Samsung CMM-H) offer 4–8× the capacity of per-node DRAM at ~300–500 ns latency — significantly lower than NVMe. When available, insert CXL as a new tier:

```
GPU (HBM)  ←→  CPU RAM  ←→  CXL Pool  ←→  NVMe SSD  ←→  Shared Storage
```

---

## XIX. Decision Matrix: Recompute vs Restore

| Condition | Prefer Restore | Prefer Recompute |
|------|------|------|
| Context length > 8K | ✓ | |
| Session resumed within 5 min | ✓ | |
| PCIe/NVLink available | ✓ | |
| Restore latency < recompute latency | ✓ | |
| Small context (<2K) | | ✓ |
| Batch-only inference | | ✓ |
| Storage bandwidth saturated | | ✓ |
| Low cache hit probability | | ✓ |

Rule of thumb:

```text
If restore_time < 0.7 * recompute_time → restore
Else → recompute
```

---

## XX. Failure Postmortem: Long-Context Chat Outage

### Incident Summary

A production assistant supporting long legal conversations experienced cascading OOM failures during peak usage.

### Symptoms Observed

- GPUs at 95% memory occupancy
- Token throughput dropped 60%
- OOMs correlated with 16K+ sessions
- Restart temporarily resolved issue

### Initial (Incorrect) Diagnosis

- Model too large
- Batch size too high
- Need more GPUs

### Root Cause (Using This Architecture)

- KV cache held in GPU memory for idle sessions
- No distinction between active decode and paused sessions
- Prefill and decode peaks overlapped
- No admission control on long-context sessions

### Corrective Actions

Mapped directly to architecture layers:

- Introduced tiered KV cache offloading (GPU → CPU → NVMe)
- Enforced idle eviction at 10 seconds
- Separated prefill and decode budgets
- Added admission control based on projected KV size
- Switched optimization target from throughput to TTFT

### Outcome

- OOMs eliminated
- Concurrent session capacity increased 3.2×
- Median TTFT improved 42%
- GPU cost per active user dropped ~35%

### Lesson

The failure was not caused by model size, but by missing memory policy. Once KV cache was treated as state, not compute, the system stabilized immediately.

---

## XXI. High Availability & Session Recovery

A production inference cluster experiences GPU node failures, OOM-triggered restarts, and rolling upgrades. Session state must survive.

### A. Session State Durability

KV cache metadata is persisted to a distributed store (etcd or Redis Cluster) on every tier transition, not just on graceful shutdown:

```python
SESSION_META_TTL_S = int(os.getenv("SESSION_META_TTL_S", "86400"))

def persist_session_meta(meta: SessionMemoryMeta):
    key = f"session_meta:{meta.tenant_id}:{meta.session_id}"
    redis_cluster.setex(key, SESSION_META_TTL_S, json.dumps(asdict(meta)))
```

GPU node crashes do not destroy session metadata — only the GPU-resident KV tensor is lost (the hot tier). The last cold copy (CPU/NVMe/shared) survives and is recoverable.

### B. Failover and Session Recovery

```python
def recover_session_after_node_failure(
    session_id: str, tenant_id: str
) -> SessionMemoryHandle:
    # Step 1: Load persisted metadata
    key = f"session_meta:{tenant_id}:{session_id}"
    raw = redis_cluster.get(key)
    if raw is None:
        # Distinguish TTL expiry from never-persisted sessions for actionable debugging
        if redis_cluster.object("idletime", key) is None:
            raise SessionNotFoundError(
                f"Session {session_id} (tenant {tenant_id}) was never persisted — "
                "ensure persist_session_meta() is called on every tier transition."
            )
        raise SessionNotFoundError(
            f"Session {session_id} (tenant {tenant_id}) metadata expired (TTL={SESSION_META_TTL_S}s). "
            "Increase SESSION_META_TTL_S or re-admit the session."
        )
    meta = SessionMemoryMeta(**json.loads(raw))

    # Step 2: Reconstruct handle from last known cold tier
    h = SessionMemoryHandle(
        meta=meta,
        kv_gpu=None,
        kv_cpu=None,
        kv_nvme_path=meta.kv_nvme_path if meta.tier == "nvme" else None,
        shared_key=meta.shared_key if meta.tier == "shared" else None,
    )

    # Step 3: Re-route to healthy node and prefetch
    target_node = cluster_router.assign_healthy_node(meta.tenant_id)
    prefetch_kv(h)   # NVMe/shared → CPU → GPU on the new node
    return h
```

Detection: heartbeat-based (Kubernetes liveness probe + control-plane health monitor). On node eviction, the scheduler marks all sessions on that node as `needs_recovery` and triggers the above flow.

### C. KV Cache Replication (Warm-Tier)

For latency-sensitive workloads, CPU-resident KV cache is optionally replicated to a secondary node:

```python
def replicate_cpu_kv(h: SessionMemoryHandle, secondary_node: str):
    if h.meta.workload_type not in ("chat", "ide"):
        return  # replication only for latency-sensitive types
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

Replication is best-effort and asynchronous — it never blocks the primary decode path.

### D. Graceful Degradation & Circuit Breakers

When GPU memory pressure is extreme and admission control is rejecting the majority of requests, the system activates a degradation cascade:

```python
class CircuitBreaker:
    def __init__(self, threshold: float = 0.85):
        self.threshold = threshold

    def check(self, gpu_memory_fraction: float):
        if gpu_memory_fraction > self.threshold:
            activate_fallback_model()       # switch to smaller / quantized model
            shed_low_priority_requests()    # return HTTP 429 + Retry-After header
            metrics.increment("circuit_breaker_open")

def shed_load(req, reason: str):
    retry_after = estimate_queue_drain_seconds()
    raise HTTPException(
        status_code=429,
        headers={"Retry-After": str(retry_after)},
        detail=f"Service under memory pressure: {reason}",
    )
```

Fallback model ladder: `70B → 13B → 7B → 3B` based on available memory headroom.

### E. Rolling Updates / Blue-Green Deployment

The memory control plane must support zero-downtime upgrades:

1. **Drain phase**: Stop scheduling new sessions on the replica being upgraded; allow in-flight decode sessions to complete.
2. **Offload phase**: Evict all GPU-resident KV caches to CPU/NVMe before upgrade begins.
3. **Upgrade**: Deploy the new control plane binary.
4. **Restore phase**: The new control plane reads metadata from Redis Cluster and resumes sessions.

```python
def drain_replica_for_upgrade(replica_id: str, timeout_s: int = 120):
    scheduler.mark_draining(replica_id)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        active = scheduler.active_sessions_on(replica_id)
        if not active:
            break
        for h in active:
            if h.meta.phase == "idle":
                offload_kv_gpu_to_cpu(h)
                persist_session_meta(h.meta)
        time.sleep(2)
    remaining = scheduler.active_sessions_on(replica_id)
    if remaining:
        logger.warning(
            f"Replica {replica_id} drain timed out with {len(remaining)} active sessions; "
            "forcing eviction"
        )
        for h in remaining:
            offload_kv_gpu_to_cpu(h)
            persist_session_meta(h.meta)
    scheduler.mark_ready_for_upgrade(replica_id)
```

---

## XXII. Scheduler Pseudocode Embedding the Decision Matrix

This scheduler treats GPU memory as a first-class resource and encodes recompute-versus-restore decisions explicitly.

```python
def schedule_request(req):
    meta = estimate_session_meta(req)

    recompute_time = estimate_recompute_time(meta)
    restore_time = estimate_restore_time(meta)

    # Decision matrix embedded
    if meta.seq_len > 8192 and restore_time < 0.7 * recompute_time:
        strategy = "restore"
    elif meta.workload_type == "batch":
        strategy = "recompute"
    elif restore_time < recompute_time:
        strategy = "restore"
    else:
        strategy = "recompute"

    # Admission control
    projected_mem = meta.kv_bytes + meta.activation_bytes_peak
    if projected_mem > free_gpu_memory():
        reject_or_queue(req)
        return

    # Execute
    if strategy == "restore":
        prefetch_kv(meta.session_id)
        wait_for_kv_ready(meta.session_id)
    else:
        invalidate_existing_cache(meta.session_id)

    dispatch_decode(req)
```

This logic ensures:
- Long-context sessions default to reuse
- Restore is only chosen when economically justified
- No request is admitted without a memory guarantee

---

## XXIII. FinOps Dashboard Specification

This dashboard translates low-level inference signals into executive-visible cost metrics.

### Core Dimensions

- Time window: hourly, daily, monthly
- Segmentation: model size, context bucket, workload type

### Key Panels

1. **Monthly GPU Spend by TTFT Tier**

```text
Spend = Σ(TTFT_seconds × GPU_cost_per_second)
```

Visual: stacked area by cache tier (GPU / CPU / NVMe / recompute)

2. **Cost per Resumed Session**

```text
Cost = restore_GPU_seconds × GPU_rate + IO_bytes × IO_rate
```

Visual: bar chart by workload type

3. **Idle KV Residency Cost**

```text
Idle Cost = idle_KV_bytes / total_GPU_memory × GPU_hourly_cost
```

Visual: heatmap by time of day

4. **Recompute Penalty**

```text
Penalty = (recompute_TTFT − restore_TTFT) × GPU_rate
```

Visual: cumulative monthly waste

5. **Capacity Efficiency**

```text
Efficiency = active_sessions / max_safe_sessions
```

Visual: gauge with SLA bands

### Alerts

- Idle KV cost > threshold
- Recompute ratio rising week-over-week
- TTFT P95 exceeding SLA

---

## XXIV. Extension: What Breaks at 64K+ Context

At 64K context, new failure modes dominate even with tiered caching.

### What Changes

- KV cache size doubles again (often >40–60 GB)
- Single-session KV cache can exceed per-GPU capacity
- Restore time approaches human-perceptible latency

### New Breakpoints

1. **Single-session dominance**
One session monopolizes GPU memory, starving others.

2. **Restore latency ceiling**
Even NVLink transfers exceed acceptable TTFT.

3. **Fragmentation amplification**
Large contiguous allocations fail despite free memory.

### Architectural Extensions

- **KV sharding across GPUs** (per-layer or per-head)
- **Paged KV cache** with windowed attention
- **Context virtualization**: only hot spans restored
- **Hierarchical summarization** to collapse cold history

### Policy Changes

```python
if meta.seq_len > 65536:
    enforce_context_windowing(meta)
    shard_kv_across_devices(meta)
```

### Economic Reality

Beyond ~64K context, the marginal cost per token rises superlinearly. Systems must trade perfect recall for sustainability.

---

## XXV. Chaos Testing & Memory Regression

Memory systems are hard to validate through happy-path integration tests alone. This section defines failure injection, load testing, and regression validation strategies.

### A. Chaos Engineering Scenarios

| Scenario | Injection Method | Expected Behavior | Pass Criteria |
|---|---|---|---|
| NVMe failure during offload | `mount --bind /dev/null /nvme/kvcache` | Fall back to CPU tier, emit metric | No OOM; error logged |
| Redis crash during shared store | `kill -9 redis-server` | Fall back to NVMe tier; no data loss | Request completes; warning logged |
| PCIe bandwidth degradation (25% nominal) | `tc qdisc` / NVIDIA MIG throttle | Restore latency ↑; scheduler switches to recompute | TTFT SLA breach < 2× |
| Sudden GPU memory spike (+20 GB in 1 s) | Inject large allocation via test hook | Admission control triggers; queue drains | Zero OOMs on active decodes |
| GPU node failure mid-decode | `kubectl delete pod --grace-period=0` | Session recovered on healthy node within TTR | Session continues, not dropped |
| Redis key expiry during active session | Set Redis TTL to 1 s in test config | Re-fetch from NVMe/CPU tier | No session data loss |

Run chaos tests in a staging environment at 70% utilization to reproduce realistic pressure.

### B. Load Testing Methodology

Validate the claimed 3.2× concurrent session improvement:

```python
import asyncio, random, time

async def synthetic_session(client, context_len: int, think_time_s: float) -> float:
    prompt = generate_synthetic_prompt(context_len)
    t0 = time.monotonic()
    async with client.stream("POST", "/v1/completions", json={"prompt": prompt}) as r:
        first_token_time = None
        async for chunk in r.aiter_text():
            if first_token_time is None:
                first_token_time = time.monotonic() - t0
    return first_token_time

async def load_test(
    concurrency: int,
    duration_s: int = 300,
    # Adjust weights to match your production context-length distribution.
    # Default profile: 30% short (<4K), 50% medium (4–32K), 20% long (>32K).
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

Compare baseline (no tiered caching) against the patched system to confirm capacity improvement. Track OOM rate, TTFT distribution, and cost per session across both configurations.

### C. Memory Regression Tests

Automated tests that assert GPU memory returns to baseline after session cleanup:

```python
import os
import torch
import pytest

# Configurable via environment variable to accommodate different GPU fragmentation behaviors
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
            f"(threshold: {GPU_LEAK_THRESHOLD_MB} MB, set GPU_LEAK_THRESHOLD_MB to adjust)"
        )

def test_session_lifecycle_no_leak():
    meta = create_test_session_meta(seq_len=8192)
    h = allocate_kv_on_gpu(meta)
    assert get_gpu_memory_mb() > 0
    offload_kv_gpu_to_cpu(h)
    torch.cuda.synchronize()
    assert h.kv_gpu is None
    cleanup_session(h.meta.session_id)
    torch.cuda.synchronize()
    # assert_no_gpu_memory_leak fixture validates memory returned to baseline
```

Run these tests on every pull request that touches memory management code paths.

### D. Canary Deployment for Policy Changes

When changing scheduler thresholds or tier policies:

1. **Deploy to canary set** (5–10% of replicas) with the new policy config.
2. **Shadow compare** TTFT P50/P95, OOM rate, and tier transition latency against control replicas.
3. **Automated gate**: If canary OOM rate > 2× control for 10 minutes, auto-rollback.
4. **Promote**: After a 30-minute clean window, roll out to the full fleet.

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

---

## XXVI. Reference vLLM Patch Sketch (Hooks and Classes)

This section shows where the architecture lands concretely inside vLLM without rewriting the engine.

### Key vLLM Touchpoints

- `SequenceGroup` → session lifecycle and workload classification
- `BlockManager` → KV cache allocation and eviction
- `Scheduler` → admission control and decode ordering

### Patch Sketch

```python
# vllm/engine/kv_cache_manager.py
class TieredKVCacheManager:
    def __init__(self, gpu_pool, cpu_pool, nvme_store):
        self.gpu_pool = gpu_pool
        self.cpu_pool = cpu_pool
        self.nvme_store = nvme_store

    def evict(self, block, reason):
        if reason == "idle":
            self.offload_gpu_to_cpu(block)
        elif reason == "pressure":
            self.offload_cpu_to_nvme(block)

    def restore(self, block):
        if block.location == "cpu":
            return self.cpu_to_gpu(block)
        if block.location == "nvme":
            return self.nvme_to_gpu(block)
```

```python
# vllm/engine/scheduler.py
class MemoryAwareScheduler(Scheduler):
    def schedule(self):
        for seq_group in self.waiting:
            meta = estimate_session_meta(seq_group)

            if not admission_control(meta):
                continue

            decision = recompute_vs_restore(meta)
            if decision == "restore":
                kv_cache_mgr.restore(seq_group.block)

            self.dispatch(seq_group)
```

```python
# vllm/engine/block_manager.py
class TieredBlockManager(BlockManager):
    def free(self, block):
        kv_cache_mgr.evict(block, reason="idle")
```

These hooks allow:
- Explicit cache residency control
- Admission control before decode
- Restore decisions driven by economics

---

## XXVII. Numerical Cost Curve at 64K+ Context

Assumptions:
- A100 80GB: $2.50/hr
- PCIe restore bandwidth: 25 GB/s
- Summarization pass cost: ~0.15 sec (7B)

### KV Cache Growth

| Context | KV Size (7B) | Restore Time | Restore Cost |
|------|------|------|------|
| 32K | ~16 GB | 0.64 s | $0.00044 |
| 64K | ~32 GB | 1.28 s | $0.00089 |
| 96K | ~48 GB | 1.92 s | $0.00133 |

### Cold Recompute Cost

| Context | Recompute Time | Cost |
|------|------|------|
| 32K | 1.2 s | $0.00083 |
| 64K | 2.4 s | $0.00167 |
| 96K | 3.6 s | $0.00250 |

### Summarization Break-Even

Summarization cost (7B):
- GPU time ≈ 0.15 s
- Cost ≈ $0.00010

**At ~64K tokens:**

```text
restore_cost ≈ summarization_cost × 9
```

Beyond this point, summarization plus windowed attention dominates both restore and recompute economically.

### H100 / H200 Restore Cost Comparison

NVLink bandwidth dramatically reduces restore latency and shifts the recompute-vs-restore trade-off:

| GPU | Bandwidth | Restore: 32K (7B) | Restore: 64K (7B) | Hourly Rate |
|---|---|---|---|---|
| A100 | PCIe ~25 GB/s | 0.64 s | 1.28 s | $2.50/hr |
| H100 | NVLink 900 GB/s | ~0.018 s | ~0.036 s | $3.50/hr |
| H200 | NVLink 900 GB/s | ~0.018 s | ~0.036 s | $4.50/hr (est.) |

On H100/H200 NVLink, restore latency drops ~35× vs PCIe A100. The restore-vs-recompute trade-off shifts dramatically — restore is almost always preferred at any context length. The `restore_ratio_threshold` (Section XXX) should be tuned to `0.95` on NVLink clusters.

### Policy

```python
if meta.seq_len > 65536:
    summarize_and_compact(meta)
```

---

## XXVIII. FinOps One-Pager (Executive View)

### Objective

Expose the cost of latency, memory, and inefficiency in dollars, not tokens.

### Core Metrics (Monthly)

1. **Cost per TTFT (P50 / P95)**

```text
$ / TTFT = Σ(GPU_seconds_to_first_token × GPU_rate)
```

2. **Idle KV Cache Cost**

```text
Idle Cost = idle_KV_bytes / total_GPU_memory × GPU_hourly_cost
```

3. **Recompute Waste**

```text
Waste = (recompute_TTFT − restore_TTFT) × GPU_rate
```

4. **Effective Concurrent Capacity**

```text
Capacity = active_sessions / max_safe_sessions
```

5. **Cost per Active User**

```text
$ / user = total_GPU_spend / monthly_active_users
```

### Red Flags

- Idle KV cost > 20% of total GPU spend
- Recompute ratio increasing week-over-week
- TTFT P95 exceeding SLA for warm resumes

### Executive Summary Line

"We reduced GPU spend per active user by 35% by treating KV cache as state, not compute."

---

## XXIX. Final Synthesis

With concrete engine hooks, numerical cost curves, and executive-level metrics, the system is complete.

- Engineers get deterministic schedulers
- Finance gets predictable unit economics
- Leadership gets measurable outcomes

This is how long-context LLM inference becomes operable at scale.

---

## XXX. Runtime Configuration & Policy Tuning

Hardcoded thresholds create operational risk: a change requires a code deployment and restart. All policy-sensitive values must be runtime-tunable without service interruption.

### A. Configuration Dataclass

```python
import os
from dataclasses import dataclass, asdict

@dataclass
class InferenceMemoryConfig:
    # Tier selection thresholds
    chat_cpu_idle_threshold_s: float = float(os.getenv("CHAT_CPU_IDLE_S", "60"))
    chat_nvme_idle_threshold_s: float = float(os.getenv("CHAT_NVME_IDLE_S", "300"))
    eviction_check_interval_s: float = float(os.getenv("EVICTION_INTERVAL_S", "10"))

    # Restore vs recompute decision
    restore_ratio_threshold: float = float(os.getenv("RESTORE_RATIO", "0.7"))

    # Context windowing
    context_windowing_threshold: int = int(os.getenv("CONTEXT_WINDOW_THRESHOLD", "65536"))

    # Cache TTLs
    shared_cache_ttl_s: int = int(os.getenv("SHARED_CACHE_TTL_S", "3600"))
    session_meta_ttl_s: int = int(os.getenv("SESSION_META_TTL_S", "86400"))

    # Circuit breaker
    circuit_breaker_memory_threshold: float = float(os.getenv("CB_MEMORY_THRESHOLD", "0.85"))

    # A/B policy variant identifier
    policy_variant: str = os.getenv("POLICY_VARIANT", "default")

_config = InferenceMemoryConfig()

def get_config() -> InferenceMemoryConfig:
    return _config

def reload_config():
    global _config
    _config = InferenceMemoryConfig()
    logger.info("Memory policy config reloaded", extra=asdict(_config))
```

Hot reload is triggered via a `SIGHUP` handler or a config-watch coroutine polling etcd/Consul. Thresholds take effect on the next scheduling cycle — no restart required.

### B. A/B Testing Framework

Run two tier policies simultaneously on different replica sets and compare outcomes:

```python
POLICY_VARIANTS = {
    "default":           {"restore_ratio_threshold": 0.70, "chat_cpu_idle_threshold_s": 60},
    "aggressive_offload": {"restore_ratio_threshold": 0.60, "chat_cpu_idle_threshold_s": 30},
    "conservative":      {"restore_ratio_threshold": 0.80, "chat_cpu_idle_threshold_s": 120},
}

def get_policy_for_replica(replica_id: str) -> dict:
    variant = replica_policy_assignment.get(replica_id, "default")
    return POLICY_VARIANTS[variant]
```

All metrics from each replica are tagged with the `policy_variant` label, enabling direct dashboard comparison:

```python
ttft_histogram.observe(
    ttft_s,
    labels={"tier": tier, "tenant_id": tenant_id, "policy": config.policy_variant},
)
oom_counter.increment(labels={"policy": config.policy_variant})
```

Compare TTFT P50/P95, OOM rate, and GPU cost per session across variants before promoting a new policy to the full fleet. Use the canary gate from Section XXV to automate promotion decisions.

