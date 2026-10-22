#!/bin/bash
# Start the local MLX model server (FinRAG's alternative to Ollama) on 127.0.0.1:8080.
#   scripts/start_mlx_server.sh              # qwen2.5-7b (default)
#   scripts/start_mlx_server.sh qwen3-8b     # Qwen3-8B, thinking mode off
# One model per server: switching models inside one server could hold two 4–5 GB
# models in memory at once, which an 8 GB Mac can't. The served model is recorded
# in data/mlx_serving.txt so the app offers exactly that one.
# The models are cached on the external drive; HF_HUB_OFFLINE=1 means nothing is
# fetched from the network at runtime (download once — see README).
MODEL_ALIAS="${1:-qwen2.5-7b}"
EXTRA=""
CACHE_LIMIT=""   # MLX buffer-cache limit in bytes; empty = MLX default
case "$MODEL_ALIAS" in
  qwen2.5-7b) REPO="mlx-community/Qwen2.5-7B-Instruct-4bit"; TEMPLATE_ARGS='{}' ;;
  # Qwen3 "thinks" step by step before answering by default — slower, and not needed
  # when the answer comes from retrieved passages.
  # Its attention (KV) cache is ~2.5x Qwen2.5-7B's per token (36 layers, 8 KV heads): on an
  # 8 GB Mac a 3,000-token prompt ran out of Metal memory, so the cache is stored in 8 bits.
  qwen3-8b)   REPO="mlx-community/Qwen3-8B-4bit"; TEMPLATE_ARGS='{"enable_thinking": false}'
              EXTRA="--kv-bits 8 --quantized-kv-start 0"
              # The GPU gets ~5.7 GB on an 8 GB Mac and the weights take 4.6 GB: MLX's cache of
              # freed buffers filled the rest after ~5 answers. Disable it (slightly slower).
              CACHE_LIMIT=0 ;;
  *) echo "unknown model '$MODEL_ALIAS' (qwen2.5-7b | qwen3-8b)" >&2; exit 2 ;;
esac
FINRAG="$(cd "$(dirname "$0")/.." && pwd)"
echo "mlx:$MODEL_ALIAS" > "$FINRAG/data/mlx_serving.txt"

export HF_HOME="/Volumes/Crucial X9/hf-cache"
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
# Single-machine use: stop MLX loading Anaconda's MPICH (it expects Open MPI and aborts).
export MLX_MPI_LIBNAME=libmpi_none.dylib
# 8 GB Macs: one request at a time, no prompt cache (FinRAG never repeats a prompt), and prefill in 512-token steps
# keep peak GPU memory down (the defaults ran out of Metal memory on a long prompt).
# mlx_lm.server has no cache-limit option, so a one-line launcher sets it before starting.
exec /opt/anaconda3/bin/python3 -c "import sys, mlx.core as mx
if '$CACHE_LIMIT': mx.set_cache_limit(int('$CACHE_LIMIT'))
sys.argv = ['mlx_lm.server'] + sys.argv[1:]
from mlx_lm.server import main; main()" \
  --model "$REPO" --host 127.0.0.1 --port 8080 --chat-template-args "$TEMPLATE_ARGS" \
  --decode-concurrency 1 --prompt-concurrency 1 --prompt-cache-size 0 --prefill-step-size 512 $EXTRA
