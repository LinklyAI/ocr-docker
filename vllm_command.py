"""Pure command builder for the embedded vLLM server."""

from __future__ import annotations


def build_vllm_command(
    *,
    model_path: str,
    model_name: str,
    port: int,
    max_model_len: str,
    gpu_memory_utilization: str,
    speculative_config: str,
    enforce_eager: bool,
    max_num_batched_tokens: str = "",
    max_num_seqs: str = "",
) -> list[str]:
    command = [
        "vllm",
        "serve",
        model_path,
        "--served-model-name",
        model_name,
        "--allowed-local-media-path",
        "/",
        "--port",
        str(port),
        "--max-model-len",
        max_model_len,
        "--gpu-memory-utilization",
        gpu_memory_utilization,
    ]
    if speculative_config.strip():
        command.extend(["--speculative-config", speculative_config])
    if enforce_eager:
        command.append("--enforce-eager")
    if max_num_batched_tokens:
        command.extend(["--max-num-batched-tokens", max_num_batched_tokens])
    if max_num_seqs:
        command.extend(["--max-num-seqs", max_num_seqs])
    return command
