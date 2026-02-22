# Code Improvement Plan

A phased plan for improving simplicity, readability, and elegance across the
transcription-tools codebase. Each phase is independently shippable — later
phases don't depend on earlier ones being merged first.

**NOTE FOR SELF: All phases complete. Item 2.6 (move TERM_CORRECTIONS/META_PHRASES to config.py) intentionally kept in cleanup.py — they're cleanup-specific implementation details, not tier/model config.**

---

## What's Already Working Well

Before the critique, credit where it's earned:

- **`config.py`** is a genuine single source of truth. Frozen dataclass +
  `MappingProxyType` for VAD params = professional-grade immutability. The tiers
  are data, not logic. This is a pattern worth keeping.
- **`cli.py`** is well-decomposed. `_run_transcription`, `_run_cleanup`,
  `_resolve_cleanup_model` each do one thing. The `run()` orchestrator reads
  like a recipe: parse → validate → transcribe → clean.
- **`text_processing.py`** is the gold standard in this codebase. Two pure
  functions, no state, thorough tests. Every module should aspire to this shape.
- **Tests are focused.** Each class maps to a single concern. Parametrize is
  used where it adds value, not everywhere.

---

## Phase 1 — Config Clarity

**Target:** `config.py`
**Problem:** `TranscriptionTier` has 16 fields, but ~8 don't apply to any given
tier. A reader seeing `vad_params` on the `slow` tier has to know it's
faster-whisper-only and mentally skip it. The dataclass carries the union of
both backends' parameters, which means every tier is padded with irrelevant
defaults.

### 1.1 Split backend-specific fields into typed dicts or nested dataclasses

The root issue: one flat dataclass serves two different backends. Each backend
needs different parameters, but today they're all mixed together.

**Current shape (16 fields, flat):**
```python
@dataclass(frozen=True)
class TranscriptionTier:
    name: str
    label: str
    backend: Literal["faster_whisper", "whisper"]
    whisper_model: str
    # faster-whisper parameters (irrelevant for "whisper" backend)
    beam_size: int = 1
    best_of: int = 1
    temperature: float = 0.0
    language: str | None = None
    vad_filter: bool = False
    vad_params: MappingProxyType | None = None
    condition_on_previous_text: bool = False
    without_timestamps: bool = True
    compute_type_gpu: str = "int8_float16"
    compute_type_cpu: str = "int8"
    # OpenAI whisper parameters (irrelevant for "faster_whisper" backend)
    initial_prompt: str | None = None
    verbose: bool = False
    fp16_on_gpu: bool = True
    # veryslow extras
    enhanced_audio: bool = False
    signal_handling: bool = False
    save_backup: bool = False
```

**Proposed shape (each tier only carries what it needs):**
```python
@dataclass(frozen=True)
class FasterWhisperParams:
    """Parameters specific to the faster-whisper (CTranslate2) backend."""
    language: str | None = None
    vad_filter: bool = False
    vad_params: MappingProxyType | None = None
    without_timestamps: bool = True
    compute_type_gpu: str = "int8_float16"
    compute_type_cpu: str = "int8"


@dataclass(frozen=True)
class OpenAIWhisperParams:
    """Parameters specific to the OpenAI whisper backend."""
    initial_prompt: str | None = None
    verbose: bool = False
    fp16_on_gpu: bool = True
    signal_handling: bool = False


@dataclass(frozen=True)
class TranscriptionTier:
    name: str
    label: str
    whisper_model: str
    backend_params: FasterWhisperParams | OpenAIWhisperParams

    # Shared across both backends
    beam_size: int = 1
    best_of: int = 1
    temperature: float = 0.0
    condition_on_previous_text: bool = False

    # Output behavior
    enhanced_audio: bool = False
    save_backup: bool = False
```

**What this achieves:**
- A reader can tell which parameters belong to which backend at a glance
- `isinstance(tier.backend_params, FasterWhisperParams)` replaces string-matching
  on `tier.backend`
- Each tier definition only lists the fields that matter to it
- `signal_handling` moves into `OpenAIWhisperParams` where it belongs (it only
  exists for the graceful-exit handler used by the OpenAI backend)

**Ripple effects:**
- `transcribe.py` dispatch changes from `if tier.backend == "faster_whisper"` to
  `if isinstance(tier.backend_params, FasterWhisperParams)`
- Each backend function reads from `tier.backend_params` instead of `tier` directly
- Tests update to match the new structure

### 1.2 Reduce tier definition verbosity

Many tier definitions explicitly set defaults that match the dataclass defaults.
For example, `veryfast` sets `beam_size=1, best_of=1, temperature=0.0` — all of
which are already the defaults. Only list what differs.

**Before:**
```python
"veryfast": TranscriptionTier(
    name="veryfast",
    label="Very Fast",
    backend="faster_whisper",
    whisper_model="tiny.en",
    beam_size=1,           # default
    best_of=1,             # default
    temperature=0.0,       # default
    language="en",
    vad_filter=False,      # default
    condition_on_previous_text=False,  # default
    without_timestamps=True,           # default
    compute_type_gpu="int8_float16",   # default
    compute_type_cpu="int8",           # default
),
```

**After:**
```python
"veryfast": TranscriptionTier(
    name="veryfast",
    label="Very Fast",
    whisper_model="tiny.en",
    backend_params=FasterWhisperParams(language="en"),
),
```

**Learning principle:** When you set a field to its default value, you're adding
noise. A reader has to compare each value against the default to know "is this
intentionally different or just explicit?" Only state what diverges from the
default. The defaults themselves document the "normal" case.

### 1.3 Consider whether `name` is redundant with the dict key

Every tier has `name="veryfast"` and lives at `TIERS["veryfast"]`. The name is
stored twice. Options:

- **Remove `name` from the dataclass** and have consumers use the dict key.
  Simplest, but means the tier doesn't know its own name.
- **Keep it** for when a tier is passed around without its key (e.g., in print
  statements). This is the current behavior and it's fine — just acknowledge
  it's intentional duplication for convenience, not an accident.

**Recommendation:** Keep `name` but add a brief comment: `# Matches the TIERS
dict key — duplicated for convenience when the tier is passed standalone.`

---

## Phase 2 — Cleanup Simplification

**Target:** `cleanup.py` (+ `text_processing.py`)
**Problem:** `TranscriptCleaner` interleaves four distinct concerns into one
class with deeply nested control flow. Reading `_process_with_adaptive_chunking`
requires holding the retry count, subdivision state, and validation result in
your head simultaneously.

### 2.1 Identify the four concerns

The class currently handles:

1. **Prompt construction** (`_build_prompt`) — pure function, no state needed
2. **API interaction** (`_call_openai`, `_handle_api_error`) — stateful (rate
   limit tracking)
3. **Response validation** (`_response_is_valid`) — pure function, no state
4. **Adaptive chunking** (`_process_with_adaptive_chunking`,
   `_split_at_word_boundaries`) — orchestration logic

These are tangled: `_process_chunk` calls the prompt builder, the API, AND the
validator. `_process_with_adaptive_chunking` calls `_process_chunk` which calls
`_handle_api_error` which mutates `self._consecutive_rate_limits`. The control
flow is:

```
clean()
  → split_into_chunks()
  → for each chunk:
      _process_with_adaptive_chunking()
        → for attempt in 1..3:
            _process_chunk()
              → _build_prompt()
              → sleep (rate-limit aware)
              → _call_openai()
                → catch → _handle_api_error() [mutates state]
              → sanitize_model_output()
              → _response_is_valid()
            if failed and can subdivide:
              → _split_at_word_boundaries()
              → for each sub:
                  _process_chunk() [recursive]
        → fallback: _apply_basic_cleanup()
```

That's 4 levels of nesting with state mutation threaded through.

### 2.2 Extract prompt building into a module-level function

`_build_prompt` doesn't use `self`. It's already a pure function hiding behind a
method. Move it out:

```python
def build_cleanup_prompt(chunk_text: str, chunk_idx: int, total: int) -> str:
    ...
```

**Learning principle:** If a method doesn't use `self`, it's a function wearing
a class costume. Free it. This makes it independently testable without
constructing a `TranscriptCleaner` (which requires an API key).

### 2.3 Extract response validation into a module-level function

Same situation — `_response_is_valid` is `@staticmethod`. It's already a
function:

```python
def is_valid_cleanup(response: str, original_word_count: int) -> bool:
    ...
```

### 2.4 Simplify the retry loop

The current retry logic is split across `_process_chunk` (handles sleep timing),
`_handle_api_error` (handles rate limit state), and
`_process_with_adaptive_chunking` (handles attempt counting and subdivision).

**Proposed shape:** A single `_process_chunk_with_retries` method that owns the
full retry lifecycle:

```python
def _process_chunk_with_retries(self, chunk: str, idx: int, total: int) -> str:
    """Try to clean a chunk, retrying and subdividing on failure."""
    for attempt in range(1, MAX_RETRIES + 1):
        result = self._try_once(chunk, idx, total, attempt)
        if result is not None:
            return result

        # Subdivide if chunk is large enough
        if len(chunk) // 2 >= MIN_SUBDIVIDE_CHARS:
            return self._process_subdivided(chunk, idx, total, attempt)

    return apply_basic_cleanup(chunk)
```

**What changed:**
- One method owns retries, one owns a single attempt, one owns subdivision
- No attempt counter threading — the for-loop IS the attempt counter
- `_apply_basic_cleanup` becomes the module-level `apply_basic_cleanup` (it's
  already a `@staticmethod`)

### 2.5 Eliminate the duplicated split logic

`TranscriptCleaner._split_at_word_boundaries` in `cleanup.py` does the same
thing as the force-split branch in `text_processing.py:split_into_chunks`. They
both split text at whitespace boundaries respecting a max character limit.

**Fix:** Add a `split_at_word_boundaries(text, max_chars)` function to
`text_processing.py` and have both call sites use it. Delete the duplicate from
`cleanup.py`.

### 2.6 Move `TERM_CORRECTIONS` and `META_PHRASES` to `config.py`

These are domain configuration, not cleanup logic. They define *what* to correct,
not *how* to correct it. Moving them to `config.py` keeps the cleanup module
focused on the *mechanism* of cleaning.

**Alternative:** Keep them in `cleanup.py` if you feel they're too
cleanup-specific to be "config." This is a judgment call. The test is: would you
ever want to change them independently of the cleanup logic? If yes → config. If
they always change together → keep co-located.

**Resolution:** Kept in `cleanup.py`. These constants are cleanup-internal
implementation details with zero external imports — moving them to `config.py`
would conflate cleanup-specific rules with tier/model configuration.

---

## Phase 3 — Transcribe Symmetry

**Target:** `transcribe.py`
**Problem:** `transcribe_faster_whisper` and `transcribe_openai_whisper` are 40
and 38 lines respectively, and they follow the exact same skeleton:

```
1. Print tier info and device
2. Load model
3. Print "Transcribing..."
4. Record start time
5. Call model.transcribe(...)
6. Extract text
7. Print elapsed time
8. Return text
```

Steps 1, 3, 4, 7, 8 are identical. Steps 2, 5, 6 differ. This is the
"same skeleton, different organs" pattern.

### 3.1 Extract timing and logging into a wrapper

```python
def _timed_transcription(tier: TranscriptionTier, device: str, fn) -> str:
    """Run a transcription function with standard logging and timing."""
    print(f"[{tier.label}] device={device}")
    print(f"Loading model '{tier.whisper_model}'...")

    start = time.time()
    text = fn()
    elapsed = time.time() - start

    print(f"Transcription completed in {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    return text
```

Then each backend becomes *just* the model-specific parts:

```python
def transcribe_faster_whisper(audio_path: str, tier: TranscriptionTier, device: str) -> str:
    from faster_whisper import WhisperModel

    compute_type = tier.compute_type_gpu if device == "cuda" else tier.compute_type_cpu
    print(f"compute_type={compute_type}")

    model = WhisperModel(tier.whisper_model, device=device, compute_type=compute_type, ...)

    def run():
        segments, info = model.transcribe(audio_path, **kwargs)
        text = " ".join(seg.text.strip() for seg in segments).strip()
        # Print realtime ratio if available
        ...
        return text

    return _timed_transcription(tier, device, run)
```

**Learning principle:** When two functions share the same skeleton, extract the
skeleton into a higher-order function (one that takes a function as an argument).
The shared parts live in one place; each variant only contains what's unique.

### 3.2 Clean up the dispatch function

The current dispatcher has a dead `else` branch:

```python
def transcribe(audio_path: str, tier: TranscriptionTier) -> str:
    device = _detect_device(backend=tier.backend)
    if tier.backend == "faster_whisper":
        return transcribe_faster_whisper(audio_path, tier, device)
    elif tier.backend == "whisper":
        return transcribe_openai_whisper(audio_path, tier, device)
    else:
        raise ValueError(f"Unknown backend: {tier.backend}")
```

If Phase 1 is implemented (backend is determined by `isinstance` on
`backend_params`), this becomes:

```python
def transcribe(audio_path: str, tier: TranscriptionTier) -> str:
    if isinstance(tier.backend_params, FasterWhisperParams):
        device = _detect_device("faster_whisper")
        return transcribe_faster_whisper(audio_path, tier, device)
    else:
        device = _detect_device("whisper")
        return transcribe_openai_whisper(audio_path, tier, device)
```

The `ValueError` branch disappears because the type system prevents it — you
can't construct a tier with an unknown backend type. **The best error handling
is making the error impossible.**

### 3.3 Simplify `_detect_device`

This function takes a `backend` string parameter, then branches on it. If Phase
1 makes the backend type-based, consider splitting into two functions:

```python
def _detect_ctranslate2_device() -> str: ...
def _detect_torch_device() -> str: ...
```

Each is half as long and doesn't need to branch. The caller already knows which
backend it's using.

---

## Phase 4 — Polish Pass

**Targets:** `audio.py`, naming consistency, test alignment

### 4.1 Flatten `convert_to_wav` error handling

**Current shape (3 layers):**
```python
try:
    subprocess.run(cmd, check=True, ...)
except subprocess.CalledProcessError as e:
    Path(tmp_path).unlink(missing_ok=True)
    raise RuntimeError(...)
except Exception:
    try:
        Path(tmp_path).unlink(missing_ok=True)
    except OSError:
        pass
    raise
finally:
    safe_input.unlink(missing_ok=True)
    try:
        safe_input.parent.rmdir()
    except OSError:
        pass
```

The `except Exception` block does the same cleanup as `CalledProcessError` but
with an extra nested try/except. The `finally` block also does cleanup.

**Proposed shape:**
```python
try:
    subprocess.run(cmd, check=True, ...)
except subprocess.CalledProcessError as e:
    detail = e.stderr.decode(errors="replace")[-1000:] if e.stderr else ""
    raise RuntimeError(f"ffmpeg failed to convert {input_path}: {detail}") from e
except Exception:
    raise
finally:
    # Always clean up temp files, regardless of success or failure
    Path(tmp_path).unlink(missing_ok=True) if not Path(tmp_path).exists() or ...
    safe_input.unlink(missing_ok=True)
    try:
        safe_input.parent.rmdir()
    except OSError:
        pass
```

Wait — the issue is that on *success* we want to keep `tmp_path` (it's the
return value). On *failure* we want to delete it. The `finally` block can't
distinguish these.

**Cleaner approach:** Use a success flag:

```python
result_path = None
try:
    subprocess.run(cmd, check=True, ...)
    result_path = Path(tmp_path)
except subprocess.CalledProcessError as e:
    detail = e.stderr.decode(errors="replace")[-1000:] if e.stderr else ""
    raise RuntimeError(f"ffmpeg failed to convert {input_path}: {detail}") from e
finally:
    safe_input.unlink(missing_ok=True)
    try:
        safe_input.parent.rmdir()
    except OSError:
        pass
    if result_path is None:
        Path(tmp_path).unlink(missing_ok=True)

return result_path
```

**What changed:**
- The generic `except Exception` block is gone — it was only doing cleanup that
  `finally` should own
- `finally` handles all temp file cleanup
- The success/failure distinction is a simple `None` check
- Added `from e` to preserve the exception chain (currently lost)

**Learning principle:** If you have `except Exception: <cleanup>; raise` right
before a `finally: <cleanup>`, the `finally` should own all the cleanup. That's
what `finally` is *for*.

### 4.2 Add `from e` to exception chains

In `audio.py:94`:
```python
raise RuntimeError(f"ffmpeg failed to convert {input_path}: {detail}")
```

Should be:
```python
raise RuntimeError(f"ffmpeg failed to convert {input_path}: {detail}") from e
```

Without `from e`, the original `CalledProcessError` traceback is lost. When
debugging, you'd see the `RuntimeError` but not the ffmpeg command that failed
or its exit code.

This also applies anywhere else `raise SomeError(...)` appears inside an
`except` block without `from`.

### 4.3 Naming consistency

A few naming patterns are inconsistent:

| Current | Issue | Suggested |
|---------|-------|-----------|
| `_copy_to_temp` | Verb is ambiguous (copy what?) | `_copy_input_to_temp` |
| `_call_openai` | Generic — could be any OpenAI call | `_send_cleanup_request` |
| `_handle_api_error` | "Handle" is vague — it returns None OR raises | `_maybe_raise_api_error` or split into two paths |
| `INTER_REQUEST_DELAY_SECONDS` | Long but clear | Fine as-is |
| `META_CHECK_PREFIX_CHARS` | What "meta" means isn't obvious without context | `COMMENTARY_CHECK_PREFIX_CHARS` |

### 4.4 Align tests with any structural changes

After each phase, update tests to match. Specific opportunities:

- **Phase 1 tests:** Test that `FasterWhisperParams` and `OpenAIWhisperParams`
  are frozen. Test that tier definitions don't carry irrelevant fields.
- **Phase 2 tests:** `build_cleanup_prompt` and `is_valid_cleanup` become
  directly testable without mocking `OpenAI`. Currently
  `TestResponseIsValid` calls `TranscriptCleaner._response_is_valid` — after
  extraction it's just `is_valid_cleanup(...)`.
- **Phase 3 tests:** No new tests needed — the dispatch logic is simpler, and
  the backend functions themselves don't change behavior.
- **Add a test for `convert_to_wav`:** Currently only `find_ffmpeg` is tested
  in `test_audio.py`. A test that mocks `subprocess.run` and verifies the
  ffmpeg command is constructed correctly would catch regressions.

---

## Execution Order

Each phase is independent and can be done in any order, but the recommended
sequence is:

```
Phase 1 (config)  →  Phase 3 (transcribe)  →  Phase 2 (cleanup)  →  Phase 4 (polish)
       ↑                     ↑                       ↑                      ↑
  Smallest blast       Depends on Phase 1       Largest change        Sweep everything
  radius, builds       for isinstance()         but self-contained    after structure
  foundation           dispatch                                       is settled
```

Phase 4 should go last because naming and error-handling polish is best done
after the structural changes settle — otherwise you're polishing code that
might move or reshape.

---

## Principles to Carry Forward

These are the recurring patterns behind the specific suggestions above:

1. **If a method doesn't use `self`, it's a function.** Free it from the class.
   This makes it independently testable and communicates "no side effects."

2. **Only state what diverges from the default.** Explicitly setting 8 fields to
   their default values is 8 lines of noise. Let defaults be defaults.

3. **Same skeleton, different organs → extract the skeleton.** When two functions
   share structure, factor out the shared part. Each variant should contain only
   what's unique to it.

4. **`finally` owns cleanup, not `except`.** If you're doing cleanup in both
   `except` and `finally`, consolidate into `finally` with a success flag.

5. **Make invalid states unrepresentable.** A type that prevents bad values is
   better than a runtime check that catches them. The `ValueError("Unknown
   backend")` branch should be impossible by construction, not by discipline.

6. **One module, one concern.** `text_processing.py` is the model —
   stateless, pure, focused. When a module mixes pure logic with I/O, stateful
   retry, and validation, it's doing too many jobs.

7. **Use `from e` in exception chains.** Always. Lost tracebacks are the #1
   source of "but what actually went wrong?" debugging sessions.
