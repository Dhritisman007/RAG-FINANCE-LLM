#!/bin/bash
# Start the local MLX model server (FinRAG's alternative to Ollama) on 127.0.0.1:8080.
# The model is cached on the external drive; HF_HUB_OFFLINE=1 means nothing is
# fetched from the network at runtime (download it once — see README).
export HF_HOME="/Volumes/Crucial X9/hf-cache"
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
# Single-machine use: stop MLX loading Anaconda's MPICH (it expects Open MPI and aborts).
export MLX_MPI_LIBNAME=libmpi_none.dylib
exec /opt/anaconda3/bin/python3 -m mlx_lm.server \
  --model mlx-community/Qwen2.5-7B-Instruct-4bit --host 127.0.0.1 --port 8080
