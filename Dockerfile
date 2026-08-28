FROM vllm/vllm-openai:v0.26.0

ARG TRANSFORMERS_VERSION=5.13.0
ARG GLM_OCR_COMMIT=cef4d0ea120d1741f5cefe8985eee45f6c8eff1d
ARG GLM_OCR_MODEL_REVISION=ca5d8b3e287e52589e37c28385d9655ee4372f9d

# git is needed for pip install from GitHub
RUN apt-get update && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/*

# Pin runtime dependencies so rebuilding this image cannot silently change them.
RUN pip uninstall -y transformers || true \
 && pip install --no-cache-dir \
      "transformers==${TRANSFORMERS_VERSION}" \
      "git+https://github.com/zai-org/glm-ocr.git@${GLM_OCR_COMMIT}" \
      "runpod==1.7.6" \
      "requests==2.33.0" \
      "pillow==12.3.0" \
 && python3 -c "from importlib.metadata import version; expected = {'transformers': '${TRANSFORMERS_VERSION}', 'glmocr': '0.1.5', 'runpod': '1.7.6', 'requests': '2.33.0', 'pillow': '12.3.0'}; actual = {name: version(name) for name in expected}; assert actual == expected, actual"

# Pre-download model weights into the image so cold starts don't hit HuggingFace
ENV HF_HOME=/root/.cache/huggingface
ENV GLM_OCR_MODEL_REVISION=${GLM_OCR_MODEL_REVISION}
RUN python3 -c "import os; from huggingface_hub import snapshot_download; snapshot_download('zai-org/GLM-OCR', revision=os.environ['GLM_OCR_MODEL_REVISION'])"
ENV HF_HUB_OFFLINE=1

# Persist vLLM compile cache on network volume to speed up cold starts
ENV VLLM_CACHE_ROOT=/runpod-volume/vllm-cache
ENV MAX_MODEL_LEN=16384
ENV GPU_MEMORY_UTILIZATION=0.95
ENV SPECULATIVE_CONFIG='{"method":"ngram","num_speculative_tokens":1,"prompt_lookup_max":1,"prompt_lookup_min":1}'
ENV ENFORCE_EAGER=0
ENV MAX_IMAGE_SIDE=2000
ENV USE_GLMOCR_SDK=1

# Force glm-ocr SDK to use local vLLM instead of MaaS.
RUN mkdir -p /root/.config/glm-ocr \
 && printf "pipeline:\n  maas:\n    enabled: false\n  ocr_api:\n    api_host: localhost\n    api_port: 8080\n" > /root/.config/glm-ocr/config.yaml

COPY handler.py /handler.py

EXPOSE 8080

ENTRYPOINT []
CMD ["python3", "-u", "/handler.py"]
