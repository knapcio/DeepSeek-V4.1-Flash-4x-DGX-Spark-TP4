# Arm64 sibling of the 0xSero pin (lmsysorg/sglang:dev-dsv41, linux/amd64
# digest sha256:c4ca651192e57e91989b5176c3665148131b9a171e53861dee87f5e57cef25b5).
ARG BASE_IMAGE=lmsysorg/sglang:dev-dsv41
FROM ${BASE_IMAGE}
WORKDIR /opt/dsv41
COPY adapter /opt/dsv41/adapter
RUN g++ -O2 -Wall -Wextra -Werror -std=c++17 -shared -fPIC -pthread \
    adapter/row_store.cpp -o adapter/librow_store.so
COPY runtime/flash_mla_sm120.py /sgl-workspace/sglang/python/sglang/kernels/ops/attention/flash_mla_sm120.py
# Fail the build if either pinned textual-placeholder anchor has drifted.
COPY runtime/patch_encoding_dsv41.py /opt/dsv41/runtime/patch_encoding_dsv41.py
RUN python3 /opt/dsv41/runtime/patch_encoding_dsv41.py
# The stock detector compiles the wrong grammar for required/named tool choice.
# Install SGLang #39665's V4.1 structural tag, refusing unknown base sources.
COPY runtime/patch_v41_tool_constraint.py /opt/dsv41/runtime/patch_v41_tool_constraint.py
COPY runtime/sglang/deepseekv41_detector.py /opt/dsv41/runtime/sglang/deepseekv41_detector.py
RUN python3 -B /opt/dsv41/runtime/patch_v41_tool_constraint.py
COPY boot.py /opt/dsv41/boot.py
COPY scripts /opt/dsv41/scripts
COPY tests /opt/dsv41/tests
COPY benchmarks /opt/dsv41/benchmarks
RUN PYTHONPATH=/opt/dsv41/adapter python3 /opt/dsv41/tests/test_anthropic_thinking.py \
 && CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 python3 -B /opt/dsv41/tests/test_v41_tool_constraint.py \
 && DSV41_PREFILL_BYPASS_SOURCE_ROOT=/sgl-workspace/sglang/python/sglang/srt/managers DSV41_PREFILL_BYPASS_EXPECT=unsupported PYTHONPATH=/opt/dsv41/adapter python3 /opt/dsv41/tests/test_prefill_bypass.py \
 && PYTHONPATH=/opt/dsv41/adapter python3 /opt/dsv41/tests/test_thinking_alias.py \
 && PYTHONPATH=/opt/dsv41/adapter python3 /opt/dsv41/tests/test_max_new_tokens.py \
 && PYTHONPATH=/opt/dsv41/adapter python3 /opt/dsv41/tests/test_loop_abort.py \
 && PYTHONPATH=/opt/dsv41/adapter python3 /opt/dsv41/tests/test_fast_load_pacing.py \
 && PYTHONPATH=/opt/dsv41/adapter python3 /opt/dsv41/tests/test_wo_a_w8.py \
 && PYTHONPATH=/opt/dsv41/adapter python3 /opt/dsv41/tests/test_draft_tau.py \
 && PYTHONPATH=/opt/dsv41/adapter python3 /opt/dsv41/tests/test_folded_fence.py \
 && PYTHONPATH=/opt/dsv41/adapter python3 /opt/dsv41/tests/test_draft_head_fp8.py \
 && BV_TEST_N=60000 PYTHONPATH=/opt/dsv41/adapter python3 /opt/dsv41/tests/test_block_verify.py \
 && PYTHONPATH=/opt/dsv41/adapter python3 /opt/dsv41/tests/test_autotune_keep.py \
 && CUDA_VISIBLE_DEVICES= PYTHONPATH=/opt/dsv41/adapter python3 /opt/dsv41/tests/test_replicated_split.py \
 && CUDA_VISIBLE_DEVICES= VC_TEST_N=60000 PYTHONPATH=/opt/dsv41/adapter python3 /opt/dsv41/tests/test_verify_cap.py \
 && cd /opt/dsv41/tests && CUDA_VISIBLE_DEVICES= PYTHONPATH=/opt/dsv41/adapter python3 test_indexer_chunked.py /sgl-workspace/sglang/python/sglang/srt/layers/attention/deepseek_v4_backend.py
ENV PYTHONPATH=/opt/dsv41/adapter \
    MODEL_PATH=/models/DeepSeek-V4.1-Flash \
    STATE_PATH=/state OFFLOAD_MODE=nvme DSV41_CACHE_GIB=16
EXPOSE 8888
# -S: skip site (the adapter's sitecustomize imports the engine, which takes >10 s on a
# busy head and marked the container unhealthy during long prefills); health needs stdlib only.
HEALTHCHECK --interval=30s --timeout=30s --start-period=30m --retries=3 \
    CMD ["python3", "-S", "/opt/dsv41/boot.py", "health"]
ENTRYPOINT ["python3", "-u", "/opt/dsv41/boot.py"]
CMD ["run"]
