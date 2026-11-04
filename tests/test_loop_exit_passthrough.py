#!/usr/bin/env python3
"""The named loop exit and the history repair change nothing for ordinary requests.

``adapter/loop_abort.py`` (named exit, MiaAI-Lab PR #55) and ``adapter/history_repair.py``
may only change two things: how a request that the loop detector stops ends (an error
instead of ``finish_reason=stop``), and a request whose history holds tool-call arguments
that stock SGLang rejects with HTTP 400. This test proves the rest is untouched:

1. Detector: ``detect_repetition`` / ``detect_repetition_detail`` decide hit / no hit and
   the kept length exactly like the v2.6.3 detector (frozen copy below) on a fuzz corpus
   (random, periodic, same-token runs, repeated lines; default and small configs).
2. Decode path: a fake ``Req`` decoded step by step (speculative accept lengths 1..6, EOS,
   length cap, a line-decoding tokenizer) with no hook, with the hook and the named exit,
   with ``DSV41_LOOP_NAMED_EXIT=0`` and with ``DSV41_LOOP_ABORT=0``: identical output ids,
   finish step, finish reason and ``finished_len`` for every request the detector does not
   stop. For requests it stops, the named and the stop exit keep the same tokens at the
   same step; only the finish reason differs.
3. History: valid tool-call arguments (JSON objects) parse to the same value, and a
   normalised history is byte-identical as JSON, with or without the repair.
4. With an SGLang python tree as ``argv[1]`` (the image build passes
   ``/sgl-workspace/sglang/python``): the pinned ``parse_tool_call_arguments`` /
   ``normalize_assistant_tool_call_arguments`` and ``encoding_dsv41.encode_messages``
   give byte-identical prompts for valid tool histories with and without the repair, the
   repaired prompt of a broken history carries the raw text, and the serving layer still
   has what the named exit relies on (``FINISH_ABORT(message, status_code, err_type)``,
   the streaming error branch for an ``HTTPStatus`` abort, the non-stream HTTP 500).
"""
import ast
import copy
import importlib.util
import json
import logging
import os
import random
import re
import sys
import types
from http import HTTPStatus
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'adapter'))

import history_repair  # noqa: E402
import loop_abort  # noqa: E402
from loop_abort import (  # noqa: E402
    MATCHED_STOP, detect_repetition, detect_repetition_detail, load_config, repeated_line_suffix)

LOOP_ENV = ('DSV41_LOOP_ABORT', 'DSV41_LOOP_NAMED_EXIT', 'DSV41_LOOP_NGRAM', 'DSV41_LOOP_NGRAM_MAX',
            'DSV41_LOOP_REPEATS', 'DSV41_LOOP_IDENTICAL', 'DSV41_LOOP_LINE_REPEATS',
            'DSV41_LOOP_LINE_MIN')


# ------------------------------------------------------------------ v2.6.3 detector, frozen
def v263_detect_repetition(ids, text=None, cfg=None):
    """adapter/loop_abort.py detect_repetition at v2.6.3 (d51d4ab), verbatim body."""
    if cfg is None:
        cfg = load_config()
    if not cfg.enabled:
        return None
    n = len(ids)
    if n == 0:
        return None

    if cfg.identical > 0 and n >= cfg.identical:
        last = ids[-1]
        run = 1
        i = n - 2
        while i >= 0 and ids[i] == last:
            run += 1
            i -= 1
        if run >= cfg.identical:
            return MATCHED_STOP, n - run + 1

    if cfg.ngram > 0 and cfg.repeats > 0:
        max_p = min(cfg.ngram_max, n // cfg.repeats)
        for period in range(cfg.ngram, max_p + 1):
            unit = ids[-period:]
            hit = True
            for r in range(1, cfg.repeats):
                start = n - (r + 1) * period
                end = n - r * period
                if ids[start:end] != unit:
                    hit = False
                    break
            if hit:
                extra = (cfg.repeats - 1) * period
                return MATCHED_STOP, n - extra

    if text is not None and repeated_line_suffix(text, cfg.line_repeats, cfg.line_min):
        return MATCHED_STOP, n
    return None


class _Env:
    def __init__(self, **kv):
        self.kv = kv

    def __enter__(self):
        self.old = {k: os.environ.get(k) for k in LOOP_ENV + ('DSV41_HISTORY_REPAIR',)}
        for k in LOOP_ENV + ('DSV41_HISTORY_REPAIR',):
            os.environ.pop(k, None)
        for k, v in self.kv.items():
            os.environ[k] = v

    def __exit__(self, *exc):
        for k, v in self.old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _configs():
    with _Env():
        default = load_config()
    small = SimpleNamespace(abort=1, ngram=3, ngram_max=12, repeats=3, identical=6,
                            line_repeats=3, line_min=4, named_exit=True, enabled=True)
    lines_only = SimpleNamespace(abort=1, ngram=0, ngram_max=0, repeats=0, identical=0,
                                 line_repeats=4, line_min=8, named_exit=True, enabled=True)
    return [default, small, lines_only]


def _corpus(rng):
    seqs = [[], [5], [7] * 63, [7] * 64, [7] * 65, list(range(40, 72)) * 3,
            list(range(40, 72)) * 4, list(range(40, 72)) * 4 + [1]]
    for _ in range(1500):
        kind = rng.randrange(5)
        if kind == 0:      # random, tiny vocabulary (hits under the small config)
            seqs.append([rng.randrange(3) for _ in range(rng.randrange(0, 120))])
        elif kind == 1:    # random, large vocabulary (the ordinary case)
            seqs.append([rng.randrange(129280) for _ in range(rng.randrange(0, 700))])
        elif kind == 2:    # a periodic tail after a random head, sometimes one copy short
            period = rng.randrange(1, 300)
            unit = [rng.randrange(50) for _ in range(period)]
            reps = rng.randrange(1, 6)
            head = [rng.randrange(129280) for _ in range(rng.randrange(0, 50))]
            seq = head + unit * reps
            if rng.random() < 0.3:
                seq = seq[:-rng.randrange(1, period + 1)]
            seqs.append(seq)
        elif kind == 3:    # a same-token run near the threshold
            run = rng.randrange(1, 80)
            seqs.append([rng.randrange(129280) for _ in range(rng.randrange(0, 30))] + [9] * run)
        else:              # an almost-periodic tail: one token changed in one copy
            period = rng.randrange(32, 80)
            unit = [rng.randrange(1000) for _ in range(period)]
            seq = unit * 4
            seq[rng.randrange(len(seq))] ^= 1
            seqs.append(seq)
    return seqs


def _texts(rng):
    out = [None, '', 'one line only', 'same line here\n' * 8, 'same line here\n' * 7,
           'short\n' * 20, 'x' * 16 + '\n' + ('the same long line\n' * 8)]
    for _ in range(200):
        lines = [rng.choice(['alpha beta gamma', 'delta epsilon zeta', 'eta']) for _ in range(rng.randrange(0, 12))]
        out.append('\n'.join(lines) + ('\n' if rng.random() < 0.5 else ''))
    return out


def test_detector_matches_v263():
    rng = random.Random(20261104)
    seqs = _corpus(rng)
    texts = _texts(rng)
    checked = hits = 0
    for cfg in _configs():
        for i, ids in enumerate(seqs):
            text = texts[i % len(texts)]
            old = v263_detect_repetition(ids, text=text, cfg=cfg)
            new = detect_repetition(ids, text=text, cfg=cfg)
            detail = detect_repetition_detail(ids, text=text, cfg=cfg)
            assert new == old, (cfg, ids[-8:], old, new)
            assert (detail is None) == (old is None), (old, detail)
            if detail is not None:
                hits += 1
                assert detail[1] == old[1], (old, detail)
                assert detail[0] in ('identical', 'ngram', 'line')
            checked += 1
    assert hits > 100 and checked - hits > 100, (checked, hits)


# ------------------------------------------------------------------ decode-path harness
EOS = 1


class FinishMatchedStr:
    def __init__(self, matched):
        self.matched = matched

    def key(self):
        return ('str', self.matched)


class FinishMatchedToken:
    def __init__(self, matched):
        self.matched = matched

    def key(self):
        return ('token', self.matched)


class FinishLength:
    def __init__(self, length):
        self.length = length

    def key(self):
        return ('length', self.length)


class FinishAbort:
    def __init__(self, message=None, status_code=None, err_type=None):
        self.message = message or 'Aborted'
        self.status_code = status_code
        self.err_type = err_type

    def key(self):
        return ('abort', self.message, self.status_code, self.err_type)


class LineTokenizer:
    """Id t decodes to the word w(t mod 1000); words divisible by 7 end a line.

    Ids that differ by a multiple of 1000 decode to the same text, so a stream can repeat
    a line without repeating a token n-gram (only the line detector can see it)."""

    def decode(self, ids, skip_special_tokens=False):
        return ''.join(f'w{t % 1000}' + ('\n' if t % 1000 % 7 == 0 else ' ') for t in ids)


def _make_module():
    class Req:
        def __init__(self, max_new_tokens):
            self.output_ids = []
            self.finished_reason = None
            self.finished_len = None
            self.max_new_tokens = max_new_tokens
            self.rid = 'r'
            self.tokenizer = LineTokenizer()

        def finished(self):
            return self.finished_reason is not None

        def update_finish_state(self, new_accepted_len=1):
            # The stock order, reduced: EOS in the accepted tokens, then the length cap.
            if self.finished():
                return
            new = self.output_ids[-new_accepted_len:]
            for j, t in enumerate(new):
                if t == EOS:
                    self.finished_reason = FinishMatchedToken(matched=EOS)
                    self.finished_len = len(self.output_ids) - len(new) + j + 1
                    return
            if len(self.output_ids) >= self.max_new_tokens:
                self.finished_reason = FinishLength(length=self.max_new_tokens)
                self.finished_len = self.max_new_tokens

    return types.SimpleNamespace(Req=Req, FINISH_MATCHED_STR=FinishMatchedStr, FINISH_ABORT=FinishAbort)


def _decode(module, stream, accept, max_new_tokens):
    req = module.Req(max_new_tokens)
    pos = step = 0
    while not req.finished() and pos < len(stream):
        n = min(accept[step % len(accept)], len(stream) - pos)
        req.output_ids.extend(stream[pos:pos + n])
        pos += n
        step += 1
        req.update_finish_state(n)
    reason = req.finished_reason.key() if req.finished_reason is not None else None
    return {'ids': list(req.output_ids), 'step': step, 'reason': reason, 'finished_len': req.finished_len}


def _variant(env):
    module = _make_module()
    if env is not None:
        with _Env(**env):
            loop_abort.install(module)
    return module


VARIANTS = {
    'stock': None,
    'named': {},
    'named_exit_0': {'DSV41_LOOP_NAMED_EXIT': '0'},
    'loop_abort_0': {'DSV41_LOOP_ABORT': '0'},
}


def _streams(rng):
    out = []
    for _ in range(300):
        n = rng.randrange(1, 900)
        s = [rng.randrange(2, 129280) for _ in range(n)]
        if rng.random() < 0.3:                      # ends on EOS somewhere
            s.insert(rng.randrange(len(s) + 1), EOS)
        out.append(('plain', s))
    for _ in range(120):                            # loops the detector must stop
        period = rng.randrange(32, 120)
        unit = [rng.randrange(2, 5000) for _ in range(period)]
        head = [rng.randrange(2, 129280) for _ in range(rng.randrange(0, 200))]
        out.append(('loop', head + unit * 8))
    for _ in range(40):                             # same-token runs
        out.append(('loop', [rng.randrange(2, 129280) for _ in range(rng.randrange(0, 100))] + [3] * 90))
    for _ in range(40):                             # a cycle that only the line detector sees
        words = [rng.randrange(20, 140) * 7 + 1 for _ in range(4)] + [7 * 11]   # 4 words, newline
        filler = [rng.randrange(2, 129280) for _ in range(rng.randrange(0, 40))]
        s = []
        for k in range(12):                         # same text per line, new ids per line
            s += [w + 1000 * (k + 1) for w in words]
        out.append(('loop', filler + s))
    return out


def test_decode_path_untouched():
    rng = random.Random(4)
    streams = _streams(rng)
    modules = {name: _variant(env) for name, env in VARIANTS.items()}
    plain = stopped = 0
    for idx, (kind, stream) in enumerate(streams):
        accept = [rng.randrange(1, 7) for _ in range(5)]
        cap = rng.choice([len(stream) + 10, max(1, len(stream) // 2), 32768])
        res = {name: _decode(m, stream, accept, cap) for name, m in modules.items()}
        named = res['named']
        hit = named['reason'] is not None and named['reason'][0] == 'abort'
        if not hit:
            plain += 1
            for name in ('stock', 'named_exit_0', 'loop_abort_0'):
                assert res[name] == named, (idx, kind, name, res[name]['reason'], named['reason'])
            continue
        stopped += 1
        stop = res['named_exit_0']
        assert stop['reason'] == ('str', MATCHED_STOP), stop['reason']
        assert stop['ids'] == named['ids'] and stop['step'] == named['step']
        assert stop['finished_len'] == named['finished_len']
        _, message, status, err_type = named['reason']
        assert isinstance(status, HTTPStatus) and status == HTTPStatus.INTERNAL_SERVER_ERROR
        assert err_type == 'loop_abort' and message.startswith('loop_abort: DSV41 repetition (art=')
        # The detector off: the request runs on exactly like stock.
        assert res['loop_abort_0'] == res['stock']
    assert plain >= 250 and stopped >= 100, (plain, stopped)


def test_without_finish_abort_falls_back_to_stop():
    module = _make_module()
    del module.FINISH_ABORT
    with _Env():
        loop_abort.install(module)
    out = _decode(module, list(range(40, 72)) * 4, [1], 32768)
    assert out['reason'] == ('str', MATCHED_STOP) and out['finished_len'] == 32


# ------------------------------------------------------------------ history repair
def stock_parse(arguments):
    """serving_chat.parse_tool_call_arguments of the pinned tree, with json for orjson."""
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError as exc:
        raise ValueError('Assistant tool call function.arguments must be valid JSON.') from exc
    if not isinstance(parsed, dict):
        raise ValueError('Assistant tool call function.arguments must be a JSON object.')
    return parsed


VALID_ARGS = [
    '{}', '{"cmd": "ls -la"}', '{"path": "/tmp/a b", "n": 3, "f": 1.5, "t": true, "z": null}',
    '{"items": [{"title": "a", "priority": "high"}, {"title": "b\\n\\"q\\""}]}',
    '{"text": "zażółć gęślą jaźń 日本語 \\u00e9"}', '  {"pad": "whitespace"}  ',
    '{"nested": {"a": {"b": {"c": [1, 2, {"d": []}]}}}}', '{"dup": 1, "dup": 2}',
]
BROKEN_ARGS = ['{"cmd": "cat b', '[]', '"x"', '3', '', '{"a": 1,}', 'null']


def test_history_valid_arguments_identical():
    mod = types.SimpleNamespace(parse_tool_call_arguments=stock_parse)
    with _Env():
        history_repair.install(mod)
    assert mod.parse_tool_call_arguments is not stock_parse
    for raw in VALID_ARGS:
        a, b = stock_parse(raw), mod.parse_tool_call_arguments(raw)
        assert a == b and json.dumps(a) == json.dumps(b), raw
    for raw in BROKEN_ARGS:
        try:
            stock_parse(raw)
        except ValueError:
            pass
        else:
            raise AssertionError(f'stock accepts {raw!r}')
        assert mod.parse_tool_call_arguments(raw) == {history_repair.REPAIR_KEY: raw}


# ------------------------------------------------------------------ the pinned SGLang tree
def _load_functions(serving_chat, parse_impl):
    """parse_tool_call_arguments + normalize_assistant_tool_call_arguments from the tree."""
    src = serving_chat.read_text()
    tree = ast.parse(src)
    want = ('parse_tool_call_arguments', 'normalize_assistant_tool_call_arguments')
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in want]
    assert sorted(n.name for n in nodes) == sorted(want), 'serving_chat lost the tool-argument functions'
    mod = types.ModuleType('pinned_serving_chat')
    mod.__dict__.update(parse_impl)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(serving_chat), 'exec'), mod.__dict__)
    return mod


def _orjson_namespace():
    import typing
    ns = {'Dict': typing.Dict, 'Any': typing.Any, 'List': typing.List, 'Optional': typing.Optional}
    try:
        import orjson
    except ImportError:
        orjson = types.SimpleNamespace(loads=json.loads, JSONDecodeError=json.JSONDecodeError)
    ns['orjson'] = orjson
    return ns


def _load_encoder(path):
    spec = importlib.util.spec_from_file_location('pinned_encoding_dsv41', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _histories():
    tools_call = lambda cid, name, args: {'id': cid, 'type': 'function', 'function': {'name': name, 'arguments': args}}
    valid = [
        [{'role': 'user', 'content': 'list the files'},
         {'role': 'assistant', 'content': '', 'tool_calls': [tools_call('c1', 'exec', '{"cmd": "ls -la", "timeout": 30}')]},
         {'role': 'tool', 'tool_call_id': 'c1', 'content': 'a.txt\nb.txt'},
         {'role': 'user', 'content': 'and now?'}],
        [{'role': 'system', 'content': 'You are terse.'},
         {'role': 'user', 'content': 'two calls'},
         {'role': 'assistant', 'content': None, 'tool_calls': [
             tools_call('c1', 'read', '{"path": "/x y/z.txt", "lines": [1, 2]}'),
             tools_call('c2', 'note', '{"text": "zażółć \\"q\\"", "pin": true, "meta": {"k": null}}')]},
         {'role': 'tool', 'tool_call_id': 'c1', 'content': 'one'},
         {'role': 'tool', 'tool_call_id': 'c2', 'content': 'ok'},
         {'role': 'user', 'content': 'thanks'}],
    ]
    broken = [{'role': 'user', 'content': 'run it'},
              {'role': 'assistant', 'content': '', 'tool_calls': [
                  tools_call('c1', 'exec', '{"cmd": "cd /work && cat a.txt ; cat b'),
                  tools_call('c2', 'exec', '{"cmd": "pwd"}')]},
              {'role': 'tool', 'tool_call_id': 'c1', 'content': 'Invalid JSON input for tool exec, please try again'},
              {'role': 'tool', 'tool_call_id': 'c2', 'content': '/work'},
              {'role': 'user', 'content': 'go on'}]
    return valid, broken


def _encode(enc, normalize, messages, mode):
    msgs = copy.deepcopy(messages)
    for m in msgs:
        normalize(m, strict=True)
    return enc.encode_messages(copy.deepcopy(msgs), thinking_mode=mode)


def check_pinned_tree(py_root):
    srt = Path(py_root) / 'sglang' / 'srt'
    serving_chat = srt / 'entrypoints' / 'openai' / 'serving_chat.py'
    encoder = srt / 'entrypoints' / 'openai' / 'encoding_dsv41.py'

    stock = _load_functions(serving_chat, _orjson_namespace())
    hooked = _load_functions(serving_chat, _orjson_namespace())
    with _Env():
        history_repair.install(hooked)
    assert hooked.parse_tool_call_arguments.__wrapped__ is not None
    for raw in VALID_ARGS:
        assert stock.parse_tool_call_arguments(raw) == hooked.parse_tool_call_arguments(raw), raw
    for raw in BROKEN_ARGS:
        try:
            stock.parse_tool_call_arguments(raw)
        except ValueError:
            pass
        else:
            raise AssertionError(f'pinned parse accepts {raw!r}')

    enc = _load_encoder(encoder)
    valid, broken = _histories()
    for mode in ('chat', 'thinking'):
        for hist in valid:
            a = _encode(enc, stock.normalize_assistant_tool_call_arguments, hist, mode)
            b = _encode(enc, hooked.normalize_assistant_tool_call_arguments, hist, mode)
            assert isinstance(a, str) and a == b, 'valid tool history encodes differently with the repair'
        try:
            _encode(enc, stock.normalize_assistant_tool_call_arguments, broken, mode)
        except ValueError:
            pass
        else:
            raise AssertionError('the pinned tree no longer rejects a broken tool-call history')
        repaired = _encode(enc, hooked.normalize_assistant_tool_call_arguments, broken, mode)
        raw = broken[1]['tool_calls'][0]['function']['arguments']
        assert f'name="{history_repair.REPAIR_KEY}" string="true">{raw}<' in repaired, repaired[-600:]
        assert 'name="cmd" string="true">pwd<' in repaired

    # What the named exit relies on in the serving layer.
    sb = ast.parse((srt / 'managers' / 'schedule_batch.py').read_text())
    cls = {n.name: n for n in sb.body if isinstance(n, ast.ClassDef)}
    assert 'FINISH_ABORT' in cls and 'FINISH_MATCHED_STR' in cls and 'Req' in cls
    init = next(f for f in cls['FINISH_ABORT'].body if isinstance(f, ast.FunctionDef) and f.name == '__init__')
    assert [a.arg for a in init.args.args] == ['self', 'message', 'status_code', 'err_type']
    to_json = ast.unparse(next(f for f in cls['FINISH_ABORT'].body
                               if isinstance(f, ast.FunctionDef) and f.name == 'to_json'))
    for key in ('"type": "abort"', '"message": self.message', '"status_code": self.status_code'):
        assert key.replace('"', "'") in to_json or key in to_json, key
    assert any(isinstance(f, ast.FunctionDef) and f.name == 'update_finish_state' for f in cls['Req'].body)

    flat = lambda p: re.sub(r'\s+', ' ', p.read_text())
    sc = flat(serving_chat)
    assert ('if finish_reason_type == "abort" and isinstance( finish_reason.get("status_code"), HTTPStatus ):'
            in sc), 'streaming error branch for an HTTPStatus abort not found'
    assert 'self.create_streaming_error_response( finish_reason.get("message", "Generation aborted."),' in sc
    tm = flat(srt / 'managers' / 'tokenizer_manager.py')
    m = re.search(r'async def _handle_abort_finish_reason\(.*?def _wait_one_response', tm)
    assert m, '_handle_abort_finish_reason not found'
    body = m.group(0)
    assert 'HTTPStatus.INTERNAL_SERVER_ERROR' in body and 'fastapi.HTTPException(' in body, body[:400]
    base = flat(srt / 'entrypoints' / 'openai' / 'serving_base.py')
    assert 'except HTTPException as e: return self.create_error_response( message=e.detail,' in base


def main():
    for name in ('loop_abort', 'history_repair'):
        logging.getLogger(name).setLevel(logging.ERROR)
    tests = [v for k, v in list(globals().items()) if k.startswith('test_')]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f'[OK]   {fn.__name__}')
        except Exception as exc:
            failed += 1
            print(f'[FAIL] {fn.__name__}: {exc!r}')
    if len(sys.argv) > 1:
        try:
            check_pinned_tree(sys.argv[1])
            print(f'[OK]   pinned SGLang tree {sys.argv[1]}')
        except Exception as exc:
            failed += 1
            print(f'[FAIL] pinned SGLang tree {sys.argv[1]}: {exc!r}')
        total = len(tests) + 1
    else:
        print('[SKIP] pinned SGLang tree (pass the SGLang python root as argv[1])')
        total = len(tests)
    print(f'{total - failed}/{total} passed')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
