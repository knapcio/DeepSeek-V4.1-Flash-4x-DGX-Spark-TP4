"""Stop a semantic generation loop before it fills the completion cap.

VIS-04 (racing-animation HTML) streamed 714k tokens / ~2.6 h. SGLang's
``--watchdog-timeout`` only fires when a step hangs; a live token stream is
healthy as far as the GPU watchdog is concerned. The 2× EXL3 recipe has no
decode-side loop detector either.

This hook wraps ``Req.update_finish_state`` and finishes the request with a
named error (``FINISH_ABORT``, HTTP 500, message starting ``loop_abort:``) when:

* the last ``repeats`` copies of an n-gram of length ``ngram``..``ngram_max``
  are identical (keeps one copy of the unit), or
* the same token is emitted ``identical`` times in a row, or
* the same decoded line (min ``line_min`` chars) appears ``line_repeats``
  times at the end of a short decode window.

A cut answer must not look finished: with ``finish_reason=stop`` a truncated
tool call reached clients as a complete one (invalid ``function.arguments``),
and the next turn was rejected with HTTP 400. ``DSV41_LOOP_NAMED_EXIT=0``
restores the old ``stop`` exit.

Defaults catch a 32-token cycle after four copies (~128 tokens, a few seconds
at 38 tok/s) without tripping on short HTML/CSS punctuation.  Set
``DSV41_LOOP_ABORT=0`` to disable.
"""
import logging
import os
from http import HTTPStatus
from types import SimpleNamespace

logger = logging.getLogger(__name__)

DEFAULT_ABORT = 1
DEFAULT_NAMED_EXIT = 1
DEFAULT_NGRAM = 32
DEFAULT_NGRAM_MAX = 256
DEFAULT_REPEATS = 4
DEFAULT_IDENTICAL = 64
DEFAULT_LINE_REPEATS = 8
DEFAULT_LINE_MIN = 16
LINE_WINDOW_TOKENS = 512
MATCHED_STOP = 'repetition'
ERR_TYPE = 'loop_abort'


def _env_int(name, default):
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    # Override site: a set DSV41_LOOP_* value that is not an int was replaced by the default with a
    # warning. A value the user set must take effect, so a bad value now stops the boot.
    try:
        return int(str(raw).strip())
    except ValueError:
        raise RuntimeError(f'{name}={raw!r}: not an int; the default {default} would replace it; '
                           f'refusing to boot') from None


def load_config():
    abort = _env_int('DSV41_LOOP_ABORT', DEFAULT_ABORT)
    ngram = _env_int('DSV41_LOOP_NGRAM', DEFAULT_NGRAM)
    ngram_max = _env_int('DSV41_LOOP_NGRAM_MAX', DEFAULT_NGRAM_MAX)
    repeats = _env_int('DSV41_LOOP_REPEATS', DEFAULT_REPEATS)
    identical = _env_int('DSV41_LOOP_IDENTICAL', DEFAULT_IDENTICAL)
    line_repeats = _env_int('DSV41_LOOP_LINE_REPEATS', DEFAULT_LINE_REPEATS)
    line_min = _env_int('DSV41_LOOP_LINE_MIN', DEFAULT_LINE_MIN)
    named_exit = _env_int('DSV41_LOOP_NAMED_EXIT', DEFAULT_NAMED_EXIT)
    # Override site: DSV41_LOOP_NGRAM_MAX below DSV41_LOOP_NGRAM is raised to DSV41_LOOP_NGRAM. A
    # user-set DSV41_LOOP_NGRAM_MAX must not be replaced, so it now stops the boot; when only the
    # default is below the user's DSV41_LOOP_NGRAM, the raise of the default is kept and logged.
    if ngram_max < ngram:
        raw_max = os.environ.get('DSV41_LOOP_NGRAM_MAX')
        if raw_max is not None and str(raw_max).strip():
            raise RuntimeError(f'DSV41_LOOP_NGRAM_MAX={raw_max!r} is below DSV41_LOOP_NGRAM={ngram}; it would '
                               f'be raised to {ngram}; refusing to boot')
        logger.warning('WARNING DSV41_LOOP_NGRAM_MAX: default %s is below DSV41_LOOP_NGRAM=%s; using %s',
                       ngram_max, ngram, ngram)
        ngram_max = ngram
    ngram_on = ngram > 0 and repeats > 0
    enabled = abort > 0 and (ngram_on or identical > 0 or line_repeats > 0)
    return SimpleNamespace(
        abort=abort,
        ngram=ngram,
        ngram_max=ngram_max,
        repeats=repeats,
        identical=identical,
        line_repeats=line_repeats,
        line_min=line_min,
        named_exit=named_exit > 0,
        enabled=enabled,
    )


def repeated_line_suffix(text, repeats, min_chars):
    """True when the last ``repeats`` non-empty lines are identical and long enough."""
    if not text or repeats <= 0 or min_chars <= 0:
        return False
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < repeats:
        return False
    last = lines[-1]
    if len(last) < min_chars:
        return False
    return all(ln == last for ln in lines[-repeats:])


def detect_repetition(ids, text=None, cfg=None):
    """Return ``(matched, finished_len)`` or ``None``."""
    hit = detect_repetition_detail(ids, text=text, cfg=cfg)
    if hit is None:
        return None
    return MATCHED_STOP, hit[1]


def detect_repetition_detail(ids, text=None, cfg=None):
    """Return ``(art, finished_len, period)`` or ``None``; art is ngram/identical/line.

    ``finished_len`` is the prefix to keep: one copy of a cycling n-gram, or
    one token of an identical run. Line hits do not trim (no token boundary).
    """
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
            return 'identical', n - run + 1, 1

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
                return 'ngram', n - extra, period

    if text is not None and repeated_line_suffix(text, cfg.line_repeats, cfg.line_min):
        return 'line', n, 0
    return None


def _decode_tail(req, ids):
    tok = getattr(req, 'tokenizer', None)
    if tok is None or not ids:
        return None
    tail = ids[-LINE_WINDOW_TOKENS:]
    # Fall-through site: a failed tokenizer.decode skips the repeated-line check
    # (DSV41_LOOP_LINE_REPEATS) for this request only; the n-gram checks still run. Kept per request,
    # logged with WARNING once per process because this runs on every decode step.
    try:
        return tok.decode(list(tail), skip_special_tokens=False)
    except Exception as exc:
        if not _DECODE_WARNED:
            _DECODE_WARNED.append(True)
            logger.warning('WARNING DSV41_LOOP_LINE_REPEATS: tokenizer.decode failed (%s); the repeated-line '
                           'check is skipped for such requests (logged once)', exc)
        return None


_DECODE_WARNED = []


def install(module):
    """Wrap ``Req.update_finish_state`` with a repetition stop."""
    cfg = load_config()
    # Override site: DSV41_LOOP_ABORT > 0 with every detector off (DSV41_LOOP_NGRAM or
    # DSV41_LOOP_REPEATS, DSV41_LOOP_IDENTICAL and DSV41_LOOP_LINE_REPEATS all <= 0) used to log
    # "disabled" and serve without the abort. A user-set DSV41_LOOP_ABORT must take effect, so this
    # now stops the boot. DSV41_LOOP_ABORT=0 still disables the feature.
    if not cfg.enabled:
        if cfg.abort > 0 and 'DSV41_LOOP_ABORT' in os.environ:
            raise RuntimeError(
                f'DSV41_LOOP_ABORT={os.environ["DSV41_LOOP_ABORT"]!r}: every detector is off (DSV41_LOOP_NGRAM='
                f'{cfg.ngram}, DSV41_LOOP_REPEATS={cfg.repeats}, DSV41_LOOP_IDENTICAL={cfg.identical}, '
                f'DSV41_LOOP_LINE_REPEATS={cfg.line_repeats}), so the abort would not run; refusing to boot')
        logger.warning('DSV41 loop abort disabled (DSV41_LOOP_ABORT=%s)', cfg.abort)
        return

    cls = module.Req
    finish_cls = module.FINISH_MATCHED_STR
    abort_cls = getattr(module, 'FINISH_ABORT', None) if cfg.named_exit else None
    # Override site: DSV41_LOOP_NAMED_EXIT=1 without FINISH_ABORT in the engine fell back to the plain
    # stop exit. start.sh always passes DSV41_LOOP_NAMED_EXIT, so a set value now stops the boot;
    # an unset value keeps the fallback with a WARNING line.
    if cfg.named_exit and abort_cls is None:
        if 'DSV41_LOOP_NAMED_EXIT' in os.environ:
            raise RuntimeError(
                f'DSV41_LOOP_NAMED_EXIT={os.environ["DSV41_LOOP_NAMED_EXIT"]!r}: the engine has no FINISH_ABORT, '
                f'so the named loop_abort exit would fall back to the stop exit; refusing to boot')
        logger.warning('WARNING DSV41_LOOP_NAMED_EXIT: FINISH_ABORT not found; falling back to the stop exit')
    original = cls.update_finish_state

    def update_finish_state(self, new_accepted_len=1):
        original(self, new_accepted_len)
        if self.finished():
            return
        ids = self.output_ids
        text = None
        if cfg.line_repeats > 0:
            text = _decode_tail(self, ids)
        hit = detect_repetition_detail(ids, text=text, cfg=cfg)
        if hit is None:
            return
        art, finished_len, period = hit
        if abort_cls is not None:
            self.finished_reason = abort_cls(
                f'{ERR_TYPE}: DSV41 repetition (art={art}, period={period}, '
                f'output_tokens={len(ids)}, kept={finished_len})',
                HTTPStatus.INTERNAL_SERVER_ERROR, ERR_TYPE)
        else:
            self.finished_reason = finish_cls(matched=MATCHED_STOP)
        self.finished_len = finished_len
        logger.warning(
            'DSV41 loop abort rid=%s matched=%s art=%s period=%s output_tokens=%s kept=%s exit=%s',
            getattr(self, 'rid', None), MATCHED_STOP, art, period, len(ids), finished_len,
            ERR_TYPE if abort_cls is not None else 'stop')

    cls.update_finish_state = update_finish_state
    logger.warning(
        'DSV41 loop abort: ngram=%s-%s x%s, identical=%s, line_repeats=%s '
        '(min %s chars), exit=%s; DSV41_LOOP_ABORT=0 disables',
        cfg.ngram, cfg.ngram_max, cfg.repeats, cfg.identical,
        cfg.line_repeats, cfg.line_min, ERR_TYPE if abort_cls is not None else 'stop')
