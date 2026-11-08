"""Install storage adapter in every serving worker, only when explicitly enabled."""
import importlib.abc
import importlib.machinery
import os
import sys

class EngramLoader(importlib.abc.Loader):
    def __init__(self, original):
        self.original = original

    def create_module(self, spec):
        return self.original.create_module(spec)

    def exec_module(self, module):
        self.original.exec_module(module)
        if module.__name__ == 'sglang.srt.layers.engram':
            from engram_backend import install
            install(module)
            # After engram_backend: Engram rows fetched on a side stream right after the
            # hasher. Gated on DSV41_ENGRAM_PREFETCH, inactive by default.
            from engram_prefetch import install as install_engram_prefetch
            install_engram_prefetch(module)
            # After engram_prefetch: the row join forks an L2 prefetch of the engram.wkv slice.
            # Gated on DSV41_L2_PREFETCH + DSV41_L2_PREFETCH_ENGRAM (adapter/l2_prefetch.py).
            if (os.environ.get('DSV41_L2_PREFETCH', '0').strip() not in ('0', 'off', 'false', '')
                    and os.environ.get('DSV41_L2_PREFETCH_ENGRAM', '0').strip() not in ('0', 'off', 'false', '')):
                from l2_prefetch import install_engram as install_l2_prefetch_engram
                install_l2_prefetch_engram(module)
        elif module.__name__ == 'sglang.srt.layers.quantization.fp8_utils':
            from mxfp8_b12x import install
            install(module)
            # AFTER b12x, so this wraps its wrapper rather than the original.
            # Gated on DSV41_SHARED_PAD_K, inactive by default.
            from shared_pad_k import install as install_shared_pad
            install_shared_pad(module)
            # Gated on DSV41_L2_PREFETCH: record which MXFP8 weights follow each RoCE collective
            # (adapter/l2_prefetch.py). Outermost wrapper; it only records the weight pointer.
            if os.environ.get('DSV41_L2_PREFETCH', '0').strip() not in ('0', 'off', 'false', ''):
                from l2_prefetch import install_fp8_utils as install_l2_prefetch_fp8
                install_l2_prefetch_fp8(module)
        elif module.__name__ == 'sglang.srt.layers.quantization.fp8':
            from mxfp8_b12x import install_fp8
            install_fp8(module)
            from shared_pad_k import install_fp8 as install_shared_pad_fp8
            install_shared_pad_fp8(module)
        elif module.__name__ == 'sglang.srt.model_executor.model_runner':
            from prefill_empty_cache import install
            install(module)
            # TEST ONLY, gated before the import: eager forward host / GPU timing (adapter/moe_launch_prepared.py)
            if os.environ.get('DSV41_MOE_LAUNCH_TIMING', '0').strip().lower() not in ('0', 'off', 'false', '', 'no'):
                from moe_launch_prepared import install_forward_timing
                install_forward_timing(module)
        elif module.__name__ == 'sglang.srt.layers.attention.deepseek_v4_backend':
            # sglang#39187 backport: dense prefill indexer scored in bounded row chunks.
            # Gate checked BEFORE the import so a disabled flag imports nothing. v1 targets
            # the dev-dsv41 image backend (self.candidate_masks); v2 is the PR verbatim for
            # candidate_metadata backends (dsv4.1 branch >= f80c91a4b). Each refuses the other.
            if os.environ.get('DSV41_INDEXER_CHUNKED', '0').strip() not in ('0', 'off', 'false', ''):
                import inspect as _inspect
                _src = _inspect.getsource(module.DeepseekV4AttnBackend._low_ratio_index_topk_dense)
                if 'self.candidate_masks' in _src:
                    from indexer_chunked import install as install_indexer_chunked
                else:
                    from indexer_chunked_v3 import install as install_indexer_chunked
                install_indexer_chunked(module)
            # Gated on DSV41_SWA_CONE_REPLAY=1: replay rows of a lost SWA resume point get their window
            # floored at the replay start and their latent / index-key writes sent to the padding slot
            # (adapter/swa_cone.py). Gate checked BEFORE the import.
            if os.environ.get('DSV41_SWA_CONE_REPLAY', '0').strip().lower() not in ('0', '', 'off', 'false', 'no'):
                from swa_cone import install_backend as install_swa_cone_backend
                install_swa_cone_backend(module)
            # DSV41_KV_LAYOUT=v41c (adapter/v41c_pool.py) / v41 (adapter/v41_pool.py, v2.7): the writer / reader call
            # sites must be the pinned ones.
            if os.environ.get('DSV41_KV_LAYOUT', '').strip() not in _V41C_OFF:
                _kv = _kv_layout_adapter()
                if _kv is not None:
                    _kv.check_backend(module)
        elif module.__name__ == 'sglang.srt.model_executor.forward_batch_info':
            # Gated on DSV41_SWA_CONE_REPLAY=1: the resident latents / index keys of a replay are copied
            # into the request's fresh pages on the forward stream (adapter/swa_cone.py).
            if os.environ.get('DSV41_SWA_CONE_REPLAY', '0').strip().lower() not in ('0', '', 'off', 'false', 'no'):
                from swa_cone import install_forward_batch as install_swa_cone_forward_batch
                install_swa_cone_forward_batch(module)
        elif module.__name__ == 'sglang.srt.mem_cache.deepseek_v4_memory_pool':
            # Only in the finder when DSV41_KV_LAYOUT is set and not v4: ratio-1/2 pools as V4.1 FP4 records (v41c),
            # or the whole V4.1 layout pair on SM12x (v41, v2.7). v41c_pool.enabled() raises on any value but v4 /
            # v41c / v41 is routed first, so a typo stops the engine instead of serving V4.
            _kv = _kv_layout_adapter()
            if _kv is not None:
                _kv.install_pool(module)
        elif module.__name__ == 'sglang.srt.model_executor.pool_configurator':
            # The same gate: pool sizing prices the ratio-1/2 records at 288 B (v41: and the latent slots at 528 B).
            _kv = _kv_layout_adapter()
            if _kv is not None:
                _kv.install_configurator(module)
        elif module.__name__ == 'sglang.srt.mem_cache.kv_cache_configurator':
            # Only in the finder with DSV41_KV_LAYOUT=v41: its by-name import of select_dsv4_kv_layout -> the wrapper.
            _kv = _kv_layout_adapter()
            if _kv is not None and hasattr(_kv, 'install_kv_configurator'):
                _kv.install_kv_configurator(module)
        elif module.__name__ == 'sglang.srt.entrypoints.openai.encoding_dsv41':
            from encoding_compat import install_encoder
            install_encoder(module)
        elif module.__name__ == 'sglang.srt.entrypoints.openai.serving_chat':
            from encoding_compat import install_serving_chat
            install_serving_chat(module)
            from history_repair import install as install_history_repair
            install_history_repair(module)
        elif module.__name__ == 'sglang.srt.entrypoints.anthropic.protocol':
            from encoding_compat import install_anthropic_protocol
            install_anthropic_protocol(module)
        elif module.__name__ == 'sglang.srt.model_loader.weight_utils':
            # Gated on DSV41_FAST_LOAD: this rank's tensors are read eagerly by a thread
            # pool instead of page-faulted through the loader's mmap (adapter/fast_load.py).
            from fast_load import install_weight_utils
            install_weight_utils(module)
        elif module.__name__ == 'sglang.srt.models.deepseek_v4':
            from fast_load import install_deepseek_v4
            install_deepseek_v4(module)
            # Gated on DSV41_WO_A_INPUT_FP8 (off by default): wo_a's input o rounded to e4m3 in 32-blocks
            # with power-of-two scales, as DeepSeek's 2026-10-12 reference does. BEFORE wo_a_w8, whose
            # mid/drop wrapper calls this one for the routes it does not round in registers itself.
            if os.environ.get('DSV41_WO_A_INPUT_FP8', '0').strip().lower() not in ('0', 'off', 'false', ''):
                from wo_a_input_fp8 import install_model as install_wo_a_input_fp8
                install_wo_a_input_fp8(module)
            # Gated on DSV41_WO_A_W8: verify/draft wo_a reads its fp8 checkpoint bytes.
            if os.environ.get('DSV41_WO_A_W8', '0').strip() not in ('0', 'off', 'false', ''):
                from wo_a_w8 import install_model as install_wo_a_w8
                install_wo_a_w8(module)
            # Gated on DSV41_VERIFY_CAP: marks target-verify forwards for the dead-row expert remap.
            if os.environ.get('DSV41_VERIFY_CAP', '').strip() not in ('', '0', 'off'):
                from verify_cap import install_model as install_verify_cap_model
                install_verify_cap_model(module)
            # Gated on DSV41_PREFILL_SP=comm|1: prefill sequence parallel (adapter/prefill_sp.py).
            # DSV41_PREFILL_SP_DEBUG alone (fingerprints on an otherwise stock boot) also installs it.
            if (os.environ.get('DSV41_PREFILL_SP', '0').strip() not in ('0', 'off', 'false', '')
                    or os.environ.get('DSV41_PREFILL_SP_DEBUG', '').strip()):
                from prefill_sp import install as install_prefill_sp
                install_prefill_sp(module)
            # Gated on DSV41_L2_PREFETCH: brackets decode/verify forwards so each RoCE collective
            # forks an L2 prefetch of the weights that follow it (adapter/l2_prefetch.py).
            if os.environ.get('DSV41_L2_PREFETCH', '0').strip() not in ('0', 'off', 'false', ''):
                from l2_prefetch import install_model as install_l2_prefetch_model
                install_l2_prefetch_model(module)
            # Gated on DSV41_CERT_HEAD: MXINT8 screening tables of the target head, built after the
            # target's load_weights (adapter/cert_head.py). Gate checked BEFORE the import.
            if os.environ.get('DSV41_CERT_HEAD', '0').strip().lower() not in ('0', 'off', 'false', '', 'no'):
                from cert_head import install_model as install_cert_head_model
                install_cert_head_model(module)
        elif module.__name__ == 'sglang.srt.models.deepseek_v4_dspark':
            # DSV41_VERIFY_CAP=conf:T needs the draft confidence head, which the engine only builds
            # in the ragged-verify modes.
            if os.environ.get('DSV41_VERIFY_CAP', '').strip().startswith('conf:'):
                from verify_cap import install_dspark as install_verify_cap_dspark
                install_verify_cap_dspark(module)
            from fast_load import install_dspark
            install_dspark(module)
            # Gated on DSV41_WO_A_INPUT_FP8: checks the draft calls the wrapped wo_a dispatch.
            if os.environ.get('DSV41_WO_A_INPUT_FP8', '0').strip().lower() not in ('0', 'off', 'false', ''):
                from wo_a_input_fp8 import install_dspark as install_wo_a_input_fp8_dspark
                install_wo_a_input_fp8_dspark(module)
            if os.environ.get('DSV41_WO_A_W8', '0').strip() not in ('0', 'off', 'false', ''):
                from wo_a_w8 import install_dspark as install_wo_a_w8_dspark
                install_wo_a_w8_dspark(module)
            # Gated on DSV41_DRAFT_MAIN_PROJ_SPLIT: the draft's replicated main_proj column-split over TP
            # (fp8 shard + all-gather; target untouched).
            if os.environ.get('DSV41_DRAFT_MAIN_PROJ_SPLIT', '0').strip() not in ('0', 'off', 'false', ''):
                from draft_main_proj import install as install_draft_main_proj
                install_draft_main_proj(module)
            # Gated on DSV41_DRAFT_HEAD_FP8: the draft's LM head from an fp8 copy (target untouched).
            if os.environ.get('DSV41_DRAFT_HEAD_FP8', '0').strip() not in ('0', 'off', 'false', ''):
                from draft_head_fp8 import install as install_draft_head_fp8
                install_draft_head_fp8(module)
            # Gated on DSV41_L2_PREFETCH + DSV41_L2_PREFETCH_DRAFT: the draft forward is bracketed
            # like the target's, so its collectives prefetch the next stage's weights.
            if (os.environ.get('DSV41_L2_PREFETCH', '0').strip() not in ('0', 'off', 'false', '')
                    and os.environ.get('DSV41_L2_PREFETCH_DRAFT', '0').strip() not in ('0', 'off', 'false', '')):
                from l2_prefetch import install_draft as install_l2_prefetch_draft
                install_l2_prefetch_draft(module)
        elif module.__name__ == 'sglang.srt.speculative.dspark_components.dspark_draft_sampler':
            # Gated on DSV41_SPEC_SYNC_FREE: rank-invariant draft noise, per-step rank-0 broadcasts
            # dropped / merged / audited (adapter/spec_sync_free.py). Gate checked BEFORE the import.
            if os.environ.get('DSV41_SPEC_SYNC_FREE', '').strip() not in ('', '0', 'off', 'false'):
                from spec_sync_free import install_sampler as install_spec_sync_free_sampler
                install_spec_sync_free_sampler(module)
            # Gated on DSV41_DRAFT_TAU (unset or 1 = off): draft proposal temperature.
            if os.environ.get('DSV41_DRAFT_TAU', '1').strip() not in ('', '1', '1.0'):
                from draft_tau import install as install_draft_tau
                install_draft_tau(module)
            # Gated on DSV41_EAGER_GLUE (adapter/eager_glue.py): stage / vcap. After draft_tau and
            # spec_sync_free: its stage cache must be the outermost stage_sampling_params.
            if os.environ.get('DSV41_EAGER_GLUE', '').strip() not in ('', '0', 'off', 'false'):
                from eager_glue import install_sampler as install_eager_glue_sampler
                install_eager_glue_sampler(module)
        elif module.__name__ == 'sglang.kernels.ops.speculative.dspark.dspark_accept':
            # Gated on DSV41_BLOCK_VERIFY: block verification for sampled rows (lossless).
            if os.environ.get('DSV41_BLOCK_VERIFY', '0').strip() not in ('0', 'off', 'false', ''):
                from block_verify import install as install_block_verify
                install_block_verify(module)
        elif module.__name__ == 'sglang.srt.speculative.dspark_components.dspark_verify':
            # Gated on DSV41_SPEC_SYNC_FREE (adapter/spec_sync_free.py). First in this branch: its
            # merge mode checks the engine's own DsparkVerifyEpilogue._accept source.
            if os.environ.get('DSV41_SPEC_SYNC_FREE', '').strip() not in ('', '0', 'off', 'false'):
                from spec_sync_free import install_verify as install_spec_sync_free_verify
                install_spec_sync_free_verify(module)
            # Gated on DSV41_FOLDED_FENCE: folded results cloned off the persistent verify buffers
            # (sglang#40919 race under overlap scheduling).
            if os.environ.get('DSV41_FOLDED_FENCE', '0').strip() not in ('0', 'off', 'false', ''):
                from folded_result_fence import install as install_folded_fence
                install_folded_fence(module)
            # Gated on DSV41_VERIFY_CAP: per-request verify length through the acceptance cutoff.
            if os.environ.get('DSV41_VERIFY_CAP', '').strip() not in ('', '0', 'off'):
                from verify_cap import install_verify as install_verify_cap_verify
                install_verify_cap_verify(module)
            # Gated on DSV41_CERT_HEAD: the certified-head flag (masked / full logits) set per verify step.
            if os.environ.get('DSV41_CERT_HEAD', '0').strip().lower() not in ('0', 'off', 'false', '', 'no'):
                from cert_head import install_verify as install_cert_head_verify
                install_cert_head_verify(module)
            # Tap for offline draft training data. Gate checked BEFORE the import, so a disabled
            # flag imports nothing. Wraps TargetVerifyExecutor.commit_hidden; capture is switched
            # at runtime by the presence of DSV41_DRAFT_CAPTURE_TRIGGER, no restart needed.
            if os.environ.get('DSV41_DRAFT_CAPTURE', '0').strip() not in ('0', 'off', 'false', ''):
                from draft_capture import install as install_draft_capture
                install_draft_capture(module)
        elif module.__name__ == 'sglang.kernels.ops.moe.moe_fused_gate':
            if os.environ.get('DSV41_VERIFY_CAP', '').strip() not in ('', '0', 'off'):
                from verify_cap import install_gate as install_verify_cap_gate
                install_verify_cap_gate(module)
        elif module.__name__ == 'sglang.srt.speculative.dspark_components.dspark_draft':
            if os.environ.get('DSV41_VERIFY_CAP', '').strip() not in ('', '0', 'off'):
                from verify_cap import install_draft as install_verify_cap_draft
                install_verify_cap_draft(module)
            if os.environ.get('DSV41_EAGER_GLUE', '').strip() not in ('', '0', 'off', 'false'):
                from eager_glue import install_draft as install_eager_glue_draft
                install_eager_glue_draft(module)
        elif module.__name__ == 'sglang.srt.speculative.dspark_components.dspark_planner':
            if os.environ.get('DSV41_VERIFY_CAP', '').strip() not in ('', '0', 'off'):
                from verify_cap import install_planner as install_verify_cap_planner
                install_verify_cap_planner(module)
        elif module.__name__ == 'sglang.srt.model_executor.runner.flashinfer_autotune':
            # Gated on DSV41_AUTOTUNE_KEEP: keep the FlashInfer autotune cache across boots under EP
            # (sglang#40320: the stock gate deletes it on every boot).
            if os.environ.get('DSV41_AUTOTUNE_KEEP', '0').strip() not in ('0', 'off', 'false', ''):
                from autotune_keep import install as install_autotune_keep
                install_autotune_keep(module)
        elif module.__name__ == 'sglang.srt.layers.linear':
            # Gated on DSV41_REPLICATED_SPLIT=<prefix suffixes>: chosen ReplicatedLinear layers
            # column-split over TP, enabled per layer only when bit-identical on every rank.
            if os.environ.get('DSV41_REPLICATED_SPLIT', '').strip():
                from replicated_split import install as install_replicated_split
                install_replicated_split(module)
        elif module.__name__ == 'sglang.srt.distributed.device_communicators.pynccl':
            # Gated on DSV41_ROCE_GATHER=<max bytes per rank>: small TP all-gathers (the draft's
            # vocab-parallel logits) take the RoCEnante one-shot kernel instead of NCCL.
            if os.environ.get('DSV41_ROCE_GATHER', '0').strip() not in ('', '0'):
                from roce_gather import install as install_roce_gather
                install_roce_gather(module)
        elif module.__name__ == 'sglang.srt.layers.quantization.mxfp4_flashinfer_cutlass_moe':
            # Gated on DSV41_MOE_B12X_NEXT: routed experts on b12x main (package b12x_next), which
            # also runs EP_SIZE=1 at TP4 (N=576 per rank). Gate checked BEFORE the import.
            if os.environ.get('DSV41_MOE_B12X_NEXT', '0').strip() not in ('0', 'off', 'false', ''):
                from moe_b12x_next import install_method as install_moe_b12x_next
                install_moe_b12x_next(module)
        elif module.__name__ == 'sglang.srt.layers.moe.moe_runner.flashinfer_cutlass':
            if os.environ.get('DSV41_MOE_B12X_NEXT', '0').strip() not in ('0', 'off', 'false', ''):
                from moe_b12x_next import install_runner as install_moe_b12x_next_runner
                install_moe_b12x_next_runner(module)
        elif module.__name__ == 'sglang.kernels.ops.layernorm.mhc':
            # Gated on DSV41_HC_FUSED: prefill-size hc mix stats in one K walk, bit-identical to
            # the stock split-K + reduce (adapter/hc_fused.py). Gate checked BEFORE the import.
            if os.environ.get('DSV41_HC_FUSED', '0').strip() not in ('0', 'off', 'false', ''):
                from hc_fused import install as install_hc_fused
                install_hc_fused(module)
        elif module.__name__ == 'sglang.srt.models.deepseek_v2':
            # Gated on DSV41_L2_PREFETCH: the bf16 router (tiny_gemm_bf16) is recorded too.
            if os.environ.get('DSV41_L2_PREFETCH', '0').strip() not in ('0', 'off', 'false', ''):
                from l2_prefetch import install_router as install_l2_prefetch_router
                install_l2_prefetch_router(module)
        elif module.__name__ in ('b12x.comm.roce.roce_oneshot', 'b12x.comm.roce_ring.roce_oneshot',
                                 'b12x.comm.roce_twoshot.roce_oneshot'):
            # Gated on DSV41_L2_PREFETCH: RoCEnante all-reduce/all-gather fork the prefetch branch.
            # Outside sglang; the finder sees its first import whichever package imports it first,
            # and install_model re-checks sys.modules in case it was imported before the finder.
            if os.environ.get('DSV41_L2_PREFETCH', '0').strip() not in ('0', 'off', 'false', ''):
                from l2_prefetch import install_roce as install_l2_prefetch_roce
                install_l2_prefetch_roce(module)
            # Gated on DSV41_ROCE_TWOSHOT=1 (which also selects this package): the per-call row
            # threshold DSV41_ROCE_TWOSHOT_ROWS of the two-shot all-reduce (adapter/roce_twoshot.py).
            if (module.__name__ == 'b12x.comm.roce_twoshot.roce_oneshot'
                    and os.environ.get('DSV41_ROCE_TWOSHOT', '0').strip() == '1'):
                from roce_twoshot import install as install_roce_twoshot
                install_roce_twoshot(module)
            # Gated on DSV41_ROCE_IDLE_PARK=1: the proxy thread parks on a futex while the scheduler
            # is idle (adapter/roce_idle_park.py); refuses on the roce / roce_ring packages.
            if os.environ.get('DSV41_ROCE_IDLE_PARK', '0').strip() == '1':
                from roce_idle_park import install_runtime as install_roce_idle_park
                install_roce_idle_park(module)
        elif module.__name__ == 'sglang.srt.layers.logits_processor':
            # Gated on DSV41_CERT_HEAD: the target-verify LM head through the certified path.
            if os.environ.get('DSV41_CERT_HEAD', '0').strip().lower() not in ('0', 'off', 'false', '', 'no'):
                from cert_head import install_logits as install_cert_head_logits
                install_cert_head_logits(module)
        elif module.__name__ == 'sglang.srt.managers.tokenizer_manager':
            # DSV41_REPLAY_GUARD (default on): a prompt-logprob request gets HTTP 400 instead of
            # raising inside the forward under decoder SWA bounded replay (all ranks exit).
            from replay_guard import install as install_replay_guard
            install_replay_guard(module)
        elif module.__name__ == 'sglang.srt.managers.scheduler':
            # DSV41_SWA_CONE_REPLAY: pin the stock PrefillAdder selection BEFORE the bypass wraps it,
            # wrap it (a first chunk shorter than the replay falls back to the stock match) AFTER.
            swa_cone_on = os.environ.get('DSV41_SWA_CONE_REPLAY', '0').strip().lower() not in ('0', '', 'off', 'false', 'no')
            if swa_cone_on:
                from swa_cone import pin_adder as pin_swa_cone_adder
                pin_swa_cone_adder()
            if os.environ.get('DSV41_PREFILL_BYPASS', '').strip():
                from prefill_bypass import install as install_prefill_bypass
                install_prefill_bypass(module)
            if swa_cone_on:
                from swa_cone import install_adder as install_swa_cone_adder
                install_swa_cone_adder()
            # Gated on DSV41_ROCE_IDLE_PARK=1: idle mark in on_idle, proxy wake before run_batch.
            if os.environ.get('DSV41_ROCE_IDLE_PARK', '0').strip() == '1':
                from roce_idle_park import install_scheduler as install_roce_idle_park_scheduler
                install_roce_idle_park_scheduler(module)
            # Gated on DSV41_BLOCK_VERIFY_FOLD: terminal fold receipts at each busy -> idle transition.
            if os.environ.get('DSV41_BLOCK_VERIFY_FOLD', '0').strip().lower() not in ('0', '', 'off', 'false', 'no'):
                from block_verify_fold import install_scheduler as install_block_verify_fold_scheduler
                install_block_verify_fold_scheduler(module)
            # Gated on DSV41_DECODE_YIELD=<pct>: running streams keep a decode share while a long
            # prompt prefills (adapter/decode_yield.py). Gate checked BEFORE the import.
            if os.environ.get('DSV41_DECODE_YIELD', '0').strip().lower() not in ('0', '', 'off', 'false', 'no'):
                from decode_yield import install as install_decode_yield
                install_decode_yield(module)
        elif module.__name__ == 'sglang.srt.managers.schedule_batch':
            from loop_abort import install as install_loop_abort
            install_loop_abort(module)
            # Gated on DSV41_SWA_CONE_REPLAY=1: a request whose SWA resume point is gone matches on full
            # KV at F - N and replays N rows exactly instead of re-prefilling (adapter/swa_cone.py).
            if os.environ.get('DSV41_SWA_CONE_REPLAY', '0').strip().lower() not in ('0', '', 'off', 'false', 'no'):
                from swa_cone import install_schedule_batch as install_swa_cone_schedule_batch
                install_swa_cone_schedule_batch(module)
            # Gated on DSV41_SWA_PROMPT_TAIL_KEEP=<tokens>: the prompt-end insert keeps the last N prompt tokens'
            # SWA live, so a revisit that diverges near the prompt end can match there (adapter/swa_prompt_tail.py;
            # wraps UnifiedRadixCache, imported from here). Gate checked BEFORE the import.
            if os.environ.get('DSV41_SWA_PROMPT_TAIL_KEEP', '0').strip().lower() not in ('0', '', 'off', 'false', 'no'):
                from swa_prompt_tail import install as install_swa_prompt_tail
                install_swa_prompt_tail(module)
        elif module.__name__ == 'sglang.srt.mem_cache.unified_radix_cache':
            # Gated on DSV41_SWA_THIN=1: superseded chunk-boundary SWA checkpoints go to the
            # evict-first end of the SWA LRU (adapter/swa_thin.py). Gate checked BEFORE the import.
            if os.environ.get('DSV41_SWA_THIN', '0').strip().lower() not in ('0', '', 'off', 'false', 'no'):
                from swa_thin import install as install_swa_thin
                install_swa_thin(module)
        else:
            # V4.1 ratio-1/2 indexers always call the FP4 DeepGEMM kernel.
            # SM120 needs its split-128 planner even when the legacy FP8
            # indexer uses the torch path. The upstream guard misses this case.
            cls = module.PagedIndexerMetadata
            original = cls.__post_init__
            def post_init(self):
                sm12 = bool(getattr(module, '_IS_SM120', False) or
                            getattr(module, '_IS_SM121', False))
                if sm12 and self.compress_ratio in (1, 2):
                    self.force_deep_gemm_metadata = True
                original(self)
            cls.__post_init__ = post_init

# DSV41_KV_LAYOUT unset or v4 (the default): the finder list is unchanged and adapter/v41c_pool.py is never imported.
_V41C_OFF = ('', 'v4')
_V41C_MODULES = (() if os.environ.get('DSV41_KV_LAYOUT', '').strip() in _V41C_OFF else
                 ('sglang.srt.mem_cache.deepseek_v4_memory_pool', 'sglang.srt.model_executor.pool_configurator')
                 + (('sglang.srt.mem_cache.kv_cache_configurator',)
                    if os.environ.get('DSV41_KV_LAYOUT', '').strip() == 'v41' else ()))


def _kv_layout_adapter():
    """The KV-layout adapter module for DSV41_KV_LAYOUT: v41 -> v41_pool (v2.7), v41c -> v41c_pool, else None / raise."""
    if os.environ.get('DSV41_KV_LAYOUT', '').strip() == 'v41':
        import v41_pool
        return v41_pool if v41_pool.enabled() else None
    import v41c_pool
    return v41c_pool if v41c_pool.enabled() else None

class EngramFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname not in ('sglang.srt.layers.engram',
                            'sglang.srt.layers.quantization.fp8_utils',
                            'sglang.srt.layers.quantization.fp8',
                            'sglang.srt.model_executor.model_runner',
                            'sglang.srt.layers.attention.deepseek_v4_backend',
                            'sglang.srt.model_executor.forward_batch_info',
                            'sglang.srt.entrypoints.openai.encoding_dsv41',
                            'sglang.srt.entrypoints.openai.serving_chat',
                            'sglang.srt.entrypoints.anthropic.protocol',
                            'sglang.srt.model_loader.weight_utils',
                            'sglang.srt.models.deepseek_v4',
                            'sglang.srt.models.deepseek_v4_dspark',
                            'sglang.srt.speculative.dspark_components.dspark_verify',
                            'sglang.srt.speculative.dspark_components.dspark_draft_sampler',
                            'sglang.kernels.ops.speculative.dspark.dspark_accept',
                            'sglang.srt.managers.schedule_batch',
                            'sglang.srt.mem_cache.unified_radix_cache',
                            'sglang.srt.managers.scheduler',
                            'sglang.srt.managers.tokenizer_manager',
                            'sglang.kernels.ops.moe.moe_fused_gate',
                            'sglang.srt.speculative.dspark_components.dspark_draft',
                            'sglang.srt.speculative.dspark_components.dspark_planner',
                            'sglang.srt.model_executor.runner.flashinfer_autotune',
                            'sglang.srt.distributed.device_communicators.pynccl',
                            'sglang.srt.layers.linear',
                            'sglang.srt.layers.quantization.mxfp4_flashinfer_cutlass_moe',
                            'sglang.srt.layers.moe.moe_runner.flashinfer_cutlass',
                            'sglang.kernels.ops.layernorm.mhc',
                            'sglang.srt.models.deepseek_v2',
                            'b12x.comm.roce.roce_oneshot',
                            'b12x.comm.roce_ring.roce_oneshot',
                            'b12x.comm.roce_twoshot.roce_oneshot',
                            'sglang.srt.layers.attention.dsv4.metadata',
                            'sglang.srt.layers.logits_processor') + _V41C_MODULES:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is not None:
            spec.loader = EngramLoader(spec.loader)
        return spec

if os.environ.get('DSV41_SOURCE'):
    sys.meta_path.insert(0, EngramFinder())

# TEST ONLY, gated on DSV41_AB_VARIANTS>=2: in-boot A/B, one CUDA graph set per flag variant
# (adapter/ab_variant.py). After the finder, before tp3_pad imports any sglang module, so every
# adapter reads the unioned gates at install. site swallows sitecustomize exceptions, which would
# leave the engine up without its adapters: any failure here kills the process instead.
if os.environ.get('DSV41_AB_VARIANTS', '').strip() not in ('', '0', '1'):
    try:
        import ab_variant
        if ab_variant.configure():
            print(ab_variant.describe(), flush=True)
    except BaseException as exc:
        print(f'DSV41_AB: refusing to start: {exc!r}', file=sys.stderr, flush=True)
        print(f'DSV41_AB: refusing to start: {exc!r}', flush=True)
        os._exit(1)

# DSV41_TOKENIZER_HEAP=trim2 (off by default): launch_server only; malloc_trim(0) after >= 256 MiB of new heap, at most
# once per 120 s, in a tokenizer idle gap or (a stream of back-to-back long prompts) at the end of a prompt's tokenization
# (adapter/heap_trim2.py). Allocation only.
if os.environ.get('DSV41_SOURCE') and os.environ.get('DSV41_TOKENIZER_HEAP', '').strip().lower() == 'trim2':
    try:
        import heap_trim2 as _heap_trim2
        _heap_trim2.install()
    except Exception as exc:
        print(f'DSV41_TOKENIZER_HEAP=trim2: not installed ({exc!r})', flush=True)

if os.environ.get('DSV41_SOURCE'):
    try:
        import tp3_pad
        tp3_pad.install()
    except Exception as exc:
        print(f'DSV41 TP pad not installed: {exc}', flush=True)
