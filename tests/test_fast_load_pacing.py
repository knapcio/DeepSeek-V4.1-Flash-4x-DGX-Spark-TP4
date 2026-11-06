"""fast_load pacing: never more than the byte budget in flight (8 bytes here, 1 per copy), all complete, sync path untouched."""
import concurrent.futures, os, sys, threading, time, types
os.environ["DSV41_FAST_LOAD"] = "1"; os.environ["DSV41_FAST_LOAD_INFLIGHT_GB"] = repr(8 / 2**30)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "adapter"))
import fast_load

def orig(*, executor, futures, use_async, func, func_args=(), func_kwargs=None):
    if use_async:
        futures.append(executor.submit(func, *func_args, **(func_kwargs or {})))
    else:
        func(*func_args, **(func_kwargs or {}))

def toy_model():
    # install_deepseek_v4 refuses a module without DeepseekV4ForCausalLM, because the release
    # before KV-pool sizing would then be skipped
    return type("DeepseekV4ForCausalLM", (), {"load_weights": lambda self, weights, *a, **k: None})

mod = types.SimpleNamespace(maybe_executor_submit=orig, DeepseekV4ForCausalLM=toy_model())
fast_load.install_deepseek_v4(mod)
assert mod.maybe_executor_submit is not orig
inflight = [0]; peak = [0]; done = [0]; lock = threading.Lock()
def work(i):
    with lock:
        inflight[0] += 1; peak[0] = max(peak[0], inflight[0])
    time.sleep(0.001)
    with lock:
        inflight[0] -= 1; done[0] += 1
futures = []
with concurrent.futures.ThreadPoolExecutor(24) as ex:
    t = time.time()
    for i in range(2000):
        mod.maybe_executor_submit(executor=ex, futures=futures, use_async=True, func=work, func_args=(i,))
    for f in concurrent.futures.as_completed(futures):
        f.result()
assert done[0] == 2000 and len(futures) == 2000, (done, len(futures))
assert peak[0] <= 8, peak
mod.maybe_executor_submit(executor=None, futures=futures, use_async=False, func=work, func_args=(0,))
assert done[0] == 2001
# an exception in the wrapped submit must not leak a permit
def boom(**kw): raise RuntimeError("x")
mod2 = types.SimpleNamespace(maybe_executor_submit=boom, DeepseekV4ForCausalLM=toy_model()); fast_load.install_deepseek_v4(mod2)
for _ in range(20):
    try: mod2.maybe_executor_submit(executor=None, futures=[], use_async=True, func=work)
    except RuntimeError: pass
print(f"pacing OK: peak in flight {peak[0]} <= 8, 2000 copies in {time.time()-t:.2f}s")
