# LLM Inference Tuning

This repository contains a production-grade architecture specification for preventing GPU out-of-memory failures in LLM inference systems. The current focus is the memory control plane design described in [`production_grade_gpu_oom_prevention_architecture_for_llm_inference_spec.md`](./production_grade_gpu_oom_prevention_architecture_for_llm_inference_spec.md).

The spec’s core thesis is that KV cache should be treated as governed, tiered, schedulable state rather than accidental runtime residue. It combines reservation-based admission control, prefill/decode isolation, and restore-vs-recompute economics to keep active decode on GPU while moving colder state through CPU, NVMe, and shared tiers. The result is a design aimed at reducing avoidable OOMs, stabilizing latency, and improving fleet efficiency.
