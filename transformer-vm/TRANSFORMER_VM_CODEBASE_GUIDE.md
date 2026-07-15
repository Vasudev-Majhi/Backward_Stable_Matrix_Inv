# Transformer-VM: Complete Codebase Guide

> **Project:** Analytically-constructed transformer that simulates a WebAssembly VM
> **Repository:** [Dhruv-Git1/transformer-vm](https://github.com/Dhruv-Git1/transformer-vm)
> **Blog:** [craft.ai/blog/constructing-llm-computer](https://www.craft.ai/blog/constructing-llm-computer)

---

## Table of Contents

1. [What This Project Does (Plain English)](#what-this-project-does-plain-english)
2. [Architecture Overview](#architecture-overview)
3. [End-to-End Pipeline](#end-to-end-pipeline)
4. [The Token Prefix Format (Critical)](#the-token-prefix-format-critical)
5. [What Runs on CPU vs What Runs in the Model](#what-runs-on-cpu-vs-what-runs-in-the-model)
6. [Directory Structure](#directory-structure)
7. [Core DSL — `graph/core.py`](#core-dsl--graphcorepy)
8. [WASM Interpreter — `wasm/interpreter.py`](#wasm-interpreter--wasminterpreterpy)
9. [Compilation Pipeline — `compilation/`](#compilation-pipeline--compilation)
10. [WASM Reference Executor — `wasm/reference.py`](#wasm-reference-executor--wasmreferencepy)
11. [MILP Scheduler — `scheduler/milp.py`](#milp-scheduler--schedulermilppy)
12. [Model & Weights — `model/`](#model--weights--model)
13. [Attention System — `attention/`](#attention-system--attention)
14. [CLI Entry Points](#cli-entry-points)
15. [Example Programs](#example-programs)
16. [Key Concepts & Insights](#key-concepts--insights)
17. [Common Confusions Clarified](#common-confusions-clarified)

---

## What This Project Does (Plain English)

Most AI models are **trained** — shown billions of examples until their weights are good enough. This project does something completely different.

It takes a C program and builds a transformer model whose weights are **mathematically calculated from scratch** — no training data, no gradient descent. The weights are derived by hand from a formal description of what a WASM virtual machine needs to compute.

The result: a tiny transformer (7 layers, 36 dimensions) that can **execute programs** by generating output tokens one at a time. Given the program as a text sequence, the model generates what the program would print — acting like a CPU whose instruction set is attention heads and FFN gates.

**The two big ideas:**
1. You can describe any computation (like "simulate a WASM VM") as a symbolic graph of linear combinations, attention lookups, and gated multiplications.
2. That symbolic graph can be mechanically converted into exact transformer weight values — no training needed.

---

## Architecture Overview

Transformer-VM compiles arbitrary C programs into transformer weights **without any gradient-based training**. The pipeline:

```
C Code → WASM bytecode → Computation Graph (DSL) → MILP Schedule → Analytical Weights → Transformer
```

Every transformer weight is derived algebraically from the computation graph. The system supports two modes:

- **Universal mode**: A single transformer that can execute _any_ WASM program (program fed as input tokens in the `{ ... }` prefix block)
- **Specialized mode** (First Futamura Projection): Program bytecode baked directly into FFN weights; model only needs the runtime input (no `{ }` block)

**Actual model dimensions (default universal build):**
| Parameter | Value |
|-----------|-------|
| `d_model` | 36 |
| `n_layers` | 7 |
| `n_heads` | 18 |
| `d_ffn` | 36 |
| Precision | float64 |

These are determined by the MILP solver — they represent the minimum size needed to fit the WASM VM computation graph.

---

## End-to-End Pipeline

```
┌────────────────────────────────────────────────────────────┐
│  C Program (e.g., collatz.c, sudoku.c)                     │
└────────────────────┬───────────────────────────────────────┘
                     │  clang --target=wasm32
                     ▼
┌────────────────────────────────────────────────────────────┐
│  WASM Binary (35 supported opcodes, hard ops lowered)      │
│  compilation/compile_wasm.py + compilation/lower.py        │
└────────────────────┬───────────────────────────────────────┘
                     │  decoder.py reads binary bytes
                     │  lower.py replaces MUL/DIV/AND/SHL etc.
                     │    with loops of ADD/SUB/branch
                     │  compile_wasm.py flattens to dispatch table
                     ▼
┌────────────────────────────────────────────────────────────┐
│  Token Prefix Text File (.txt)                             │
│  { i32.const 48 00 00 00  output 00 00 00 00  halt ... }  │
│  + input bytes + commit token                              │
└────────────────────┬───────────────────────────────────────┘
                     │  fed to transformer as starting context
                     ▼
┌────────────────────────────────────────────────────────────┐
│  Transformer Model (model.bin)                             │
│  generates output tokens one at a time:                    │
│  out(H) out(e) out(l) out(l) out(o) ...                   │
└────────────────────────────────────────────────────────────┘
```

**One-time setup (build the model):**
```
wasm/interpreter.py → computation graph blueprint
scheduler/milp.py   → which layer does what
model/weights.py    → exact numbers for every weight
                    → saved as model.bin
```

**Every run (execute a program):**
```
compilation/        → C file to token prefix .txt
runner.py           → feed .txt to model.bin → output tokens
wasm/reference.py   → run same program directly → compare
```

---

## The Token Prefix Format (Critical)

**The `.txt` file is NOT just a text version of the WASM binary.** Major transformations happen. Understanding this is essential.

### What changes at each step

| Step | What it produces | What changes |
|------|-----------------|--------------|
| C → WASM | Binary `.wasm` file | clang compiles; result is machine code for a stack machine |
| WASM → Python objects | `WasmModule` | Same semantics, just readable Python; nothing semantic changes |
| Python → lower.py → Python | Modified `WasmModule` | **Big change:** hard ops replaced with loops of simple ops. `i32.mul` → 20-30 ADD instructions |
| Python → compile_wasm.py → `.txt` | Token prefix text | **Big change:** nested block/loop/if structure → flat jump table; all functions merged into one table; globals → memory reads at address 8+ |

### What the token prefix looks like

```
{
input_base 00 10 00 00
i32.const 00 00 00 00
local.set 00 00 00 00
i32.const 48 00 00 00
i32.const 00 00 00 00
i32.store8 00 00 00 00
...
output 00 00 00 00
halt 00 00 00 00
}
W o r l d 00 commit(+0,sts=0,bt=0)
```

**Structure:**
- `{` ... `}` block: the compiled program instructions. Each instruction is 5 tokens: `opcode b0 b1 b2 b3` (opcode name + 4 hex bytes for the immediate value)
- After `}`: the program's runtime input as individual character tokens or 2-digit hex escapes, then a null byte `00`, then `commit(...)` 
- The `commit(...)` token is the signal: "the program prefix is done, now start executing"

**Why each instruction needs 4 bytes:** The immediate value (like a branch target or constant) is stored as 4 little-endian bytes. So `i32.const 5` becomes `i32.const 05 00 00 00`.

### Control flow flattening

WASM uses nested structured blocks. The `.txt` uses flat jumps:
```
WASM:                          TXT:
  block                          local.get 0  
    loop                         i32.eqz
      local.get 0                br_if 12 00 00 00   ← "jump to position 12"
      i32.eqz
      br_if 1    ← "exit block"  ...
      ...
    end
  end
```

### Globals are lowered to memory

WASM global variables → memory reads/writes at `GLOBAL_BASE = 8`:
- Global 0 lives at address 8
- Global 1 lives at address 12
- `global.get 0` → `i32.const 8; i32.load`
- `global.set 0` → save to temp local; `i32.const 8; local.get temp; i32.store`

---

## What Runs on CPU vs What Runs in the Model

```
CPU (Python/C++):
  hello.c
    │ clang compiles
  hello.wasm  (binary)
    │ decoder.py reads bytes
  WasmModule  (Python objects)
    │ lower.py replaces hard ops
  WasmModule  (simplified instructions only)
    │ compile_wasm.py flattens + links
  hello.txt   (token prefix text file)

Model (GPU/CPU):
  Reads all tokens of hello.txt one at a time (using KV cache)
  After the last token (commit):
    Generates one new token per forward pass
    H → e → l → l → o → ! 
    Until halt token is generated
```

**The model never sees the C file or the WASM binary.** It only reads the `.txt` file. The compilation is purely a CPU pre-processing step.

The KV cache (from the `attention/` module) is what makes this efficient — the model doesn't re-read the full program prefix for every generated token. It caches the keys and values and does one O(log n) lookup per generated token.

---

## Directory Structure

```
transformer-vm/
├── pyproject.toml                    # Dependencies, CLI entry points
├── README.md
├── LICENSE                           # Apache-2.0
├── .pre-commit-config.yaml           # Ruff linter/formatter
├── .github/                          # CI workflows
├── assets/                           # Documentation images
└── transformer_vm/                   # Main package (~9,100 lines)
    ├── graph/
    │   ├── core.py                   # DSL primitives (Expression, ReGLU, Persist, LookUp)
    │   └── __init__.py               # Public API exports
    ├── wasm/
    │   ├── interpreter.py            # WASM VM described as computation graph
    │   ├── reference.py              # Ground-truth Python WASM executor (answer key)
    │   └── __init__.py
    ├── compilation/
    │   ├── compile_wasm.py           # C → WASM → token prefix pipeline
    │   ├── decoder.py                # WASM binary format parser
    │   ├── lower.py                  # Hard-op lowering (MUL/DIV/AND/SHL → ADD/SUB chains)
    │   └── runtime.h                 # C runtime injected into compiled programs (no stdlib)
    ├── scheduler/
    │   └── milp.py                   # MILP solver for optimal layer scheduling
    ├── model/
    │   ├── weights.py                # Analytical weight construction
    │   ├── transformer.py            # PyTorch VanillaTransformer (75 lines)
    │   └── transformer.cpp           # Standalone C++ inference engine
    ├── attention/
    │   ├── hull2d_cht.h              # O(log n) 2D convex hull hard attention (C++)
    │   ├── hull_ext.cpp              # pybind11 bindings for hull cache
    │   ├── hull_cache.py             # Python wrapper for hull KV cache
    │   └── standard_cache.py         # O(n) softmax reference cache
    ├── build.py                      # Entry point: build universal transformer
    ├── runner.py                     # Entry point: run programs through transformer
    ├── evaluator.py                  # Entry point: exact graph evaluation (no weights)
    ├── specialize.py                 # Entry point: First Futamura Projection
    ├── examples/
    │   ├── hello.c, collatz.c, fibonacci.c, addition.c
    │   ├── min_cost_matching.c, sudoku.c, lowering_test.c
    │   └── manifest.yaml             # Program list with default arguments
    └── tests/
        ├── test_smoke.py             # End-to-end pipeline tests
        ├── test_distill.py           # Model building & inference tests
        └── test_specialize.py        # Futamura projection tests
```

---

## Core DSL — `graph/core.py`

**Purpose:** A blueprint language for describing computations that can be converted into transformer weights. Think of it as algebra — you write formulas using named variables, and the system later turns those formulas into actual weight matrix rows.

**The key metaphor:** The transformer has a residual stream — a vector of numbers flowing through every layer. Each slot in this vector is a named `Dimension`. The DSL lets you express relationships between slots using math, attention lookups, and gated multiplications.

### Global Constants

| Name | Value | Purpose |
|------|-------|---------|
| `BIG` | `1e30` | Large constant to zero-out attention keys via clear_key mechanism |
| `KEY_OFFSET` | `0` | Offset for attention key numerical stability |
| `LATEST_ALPHA` | `0.3` | Tie-break weight favoring recent tokens in hardmax attention |

### Global State

| Variable | Purpose |
|----------|---------|
| `_all_dims` | Accumulates all Dimension instances during graph construction |
| `_all_lookups` | Accumulates all LookUp instances during graph construction |
| `_multiply_cache`, `_reglu_cache`, `_stepglu_cache`, `_clear_key_cache` | Deduplication caches — same expression created twice returns the same object |

### Classes

#### `Expression` — A Math Formula Over Slots

Linear combination of Dimensions with numeric coefficients. The fundamental building block.

```python
expr = 2 * stack_pointer + cursor - 3 * call_depth
# Internally: {stack_pointer: 2, cursor: 1, call_depth: -3}
```

| Method | What it does |
|--------|-------------|
| `__add__`, `__sub__`, `__mul__` | Arithmetic with Expressions, Dimensions, or scalars |
| `evaluate(values)` | Evaluate expression given a dict of dimension → float value |
| `copy()` | Independent deep copy |

#### `Dimension` — A Named Slot in the Residual Stream

Base class for symbolic variables. Each Dimension will eventually be assigned to one specific index in the model's d_model vector.

```python
stack_pointer = Dimension("stack_pointer")  # declares a slot, no value yet
```

#### `InputDimension` — Always-Available Values

Four built-in instances injected through positional encoding at every token position:

| Name | What it holds |
|------|--------------|
| `one` | Constant 1 (always 1 at every position) |
| `position` | Token index (0, 1, 2, 3, ...) |
| `inv_log_pos` | `1/ln(2) - 1/ln(pos+2)` — small number that decreases as position grows |
| `position_sq` | `position²` — needed for 2D key encoding |

These are in slots 0, 1, 2 (fixed). `add_position_encoding()` in `transformer.py` injects them.

#### `ReGLUDimension` — A Conditional Gate (Non-linearity)

```python
ReGLUDimension(a_expr, b_expr)
# Computes: ReLU(b_expr) * a_expr
# = a_expr * b_expr  if b_expr >= 0
# = 0                if b_expr < 0
```

This is how the transformer does if/else. Maps to one neuron pair in the FFN:
- `b_expr` → the gate row (condition)
- `a_expr` → the value row (content)

#### `PersistDimension` — Store a Computed Value

```python
persist(2 * a + b)  # compute this expression and save result in a dedicated slot
```

Without persist, every later layer that needs `2*a + b` recomputes it. With persist it's saved once and read directly. Like a temporary variable. Also used for slot reuse: once persisted, the constituent dimensions can be freed.

#### `LookUp` / `LookUpDimension` — An Attention Head

One `LookUp` = one attention head. Answers: "look back through all past tokens, find the one where `key` best matches `query`, and return its `value`."

```python
cursor = fetch(
    value = cursor_dim,      # "I want to read cursor"
    query = position,        # "from my current position"
    key   = position - 1,   # "find the token at position-1"
)
```

Keys and queries are 2D (required by the convex hull attention trick).

#### `CumSumDimension` — Running Sum

Cumulative sum via attention averaging × position. The `fetch_sum()` function implements this.

#### `ProgramGraph` — The Finished Blueprint

Immutable snapshot of the complete computation graph. Contains all dims and lookups. Passed to the scheduler and weight builder.

### Core API Functions

| Function | Signature | What it does |
|----------|-----------|-------------|
| `reglu(a, b)` | → Expression | `ReLU(b) * a` — single FFN neuron. Use when b is always ≥ 0 |
| `stepglu(a, b)` | → Expression | `a * step(b ≥ 0)` — equals `a` when b≥0, else 0. Two ReGLU + persist internally |
| `persist(expr)` | → Expression | Materialize expression into dedicated residual slot |
| `fetch(value, query, key, clear_key, tie_break)` | → Dimension(s) | Attention-based lookup; maps 1D key/query to 2D internally |
| `fetch_sum(value_list)` | → Expression(s) | Cumulative sum: `avg(value) * position` over all past tokens |
| `auto_name(locals())` | → None | Name graph nodes from Python variable names for debugging |
| `reset_graph()` | → None | Clear all global state; recreate built-in dims. Call before building a new graph |

### Key Internals

**`_to_2d_key(k)`**: Maps 1D key to 2D: `kx = 2k`, `ky = -k²`.
The dot product `q·k = qx*kx + qy*ky = (q)(2k) + (1)(-k²) = -(k-q)² + q²`.
So maximizing `q·k` = finding the key closest to query on the number line.

**`_to_2d_query(q)`**: Maps 1D query to 2D: `[q - KEY_OFFSET, 1]`

**`_expr_key(expr)`**: Creates hashable cache key from Expression terms for deduplication.

**`clear_key` mechanism**: Setting `clear_key = some_expr` adds `-BIG * some_expr` to `ky`, making that key's score -∞ when `some_expr > 0`. Used to invalidate stale memory writes.

---

## WASM Interpreter — `wasm/interpreter.py`

**Purpose:** The person who wrote the blueprint. This file uses the DSL from `graph/core.py` to formally describe every operation a WASM VM needs to perform. It produces the computation graph that the scheduler and weight builder then use.

**This is the brain of the whole system.** It answers: "what must the transformer compute at each step to correctly simulate a WASM virtual machine?"

### Circle-Point Opcode Dispatch

Each of the 35 opcodes maps to a unique 2D point on a circle where `x² + y² = 32045`:
```
"i32.add" → (166, 67)
"i32.sub" → (166, -67)
"br_if"   → (178, -19)
...
```

To detect "is the current opcode X?":
```python
op_dot(op) = px * fetched_opcode_x + py * fetched_opcode_y - 32045 + 1
# = 1 when opcode matches (dot product equals radius² → point is on the circle)
# ≤ -1 otherwise
is_op(op) = reglu(one, op_dot(op))  # activates only for matching opcode
```

One ReGLU neuron per opcode. No switch-case needed.

### Supported Opcodes

| Category | Opcodes |
|----------|---------|
| **Control** | `halt`, `return`, `call`, `br`, `br_if` |
| **Stack** | `drop`, `select` |
| **Variables** | `local.get/set/tee`, `global.get/set` |
| **Memory** | `i32.load`, `i32.load8_s/u`, `i32.load16_s/u`, `i32.store`, `i32.store8`, `i32.store16` |
| **Constants** | `i32.const` |
| **Arithmetic** | `i32.add`, `i32.sub` |
| **Comparison** | `i32.eqz`, `i32.eq`, `i32.ne`, `i32.lt_s/u`, `i32.gt_s/u`, `i32.le_s/u`, `i32.ge_s/u` |
| **I/O** | `output`, `input_base` |

**Not supported** (lowered at compile time in `lower.py`): MUL, DIV, MOD, AND, OR, XOR, SHL, SHR, ROTL, ROTR, CLZ, CTZ, POPCNT.

### `build(program=None)` — Two Modes

- `program=None` (universal): model reads opcodes from the token prefix. Instruction fetch uses `fetch()` attention.
- `program=[list of instructions]` (specialized): opcodes are baked into FFN via `_cursor_lookup()` piecewise-constant ReGLU pairs. No program prefix needed at inference time.

### Section-by-Section Breakdown

**1. Opcode Detection**
- `op_dot(op)`: dot product with circle point → 1 if match, ≤-1 otherwise
- `is_op(op)`: `reglu` gate (universal) or `stepglu` gate (specialized)

**2. Input Dimensions**
- Always present: `byte_number`, `carry`, `delta_cursor`, `delta_stack`, `is_jump`, `store_to_stack`, `is_branch_taken`, `delta_call_depth`
- Universal only: `delta_stack_prefix`, `store_to_stack_prefix`, `opcode_x`, `opcode_y`, `is_write`

**3. Token Dictionary — What Each Token Carries**

| Token type | Example | What it encodes in its embedding |
|------------|---------|----------------------------------|
| Byte tokens | `"7f"`, `"7f'"` | `(byte_value + 1) * byte_number + carry_bit * carry` |
| Commit tokens | `"commit(+1,sts=1,bt=0)"` | `delta_cursor + stack_delta * delta_stack + sts * store_to_stack + bt * is_jump` |
| Output tokens | `"out(H)"`, `"out(0a)"` | `delta_cursor * 1` |
| Special | `"branch_taken"` | `1 * is_branch_taken` |
| Special | `"call_commit"` | `delta_cursor + delta_call_depth + is_jump` |
| Special | `"return_commit"` | `delta_cursor - delta_call_depth + is_return_commit + is_jump` |
| Program start | `"{"` (universal) / `"start"` (specialized) | 0 |
| Opcode tokens | `"i32.add"` | `px * opcode_x + py * opcode_y + sd * delta_stack_prefix + ...` |

Every non-start token also has `one = 1` baked in (used as a constant).

**4. Store Value & Branch Offset**
- Fetches 4 successive byte tokens to reconstruct a 32-bit immediate
- `store_value = sum of fetched bytes * position weights`
- Branch offset sign-extended using `jump_sign = stepglu(one, msb + 128*is_jump - 256)`

**5. Cumulative State — The Running Counters**
```python
stack_depth, cursor, call_depth = fetch_sum([delta_stack, delta_cursor_expr, delta_call_depth])
```
These are the three most important state variables. `fetch_sum` computes the running sum of each delta across all past tokens — giving the current stack depth, program counter, and call nesting level at any point.

**6. Instruction Fetch**
- **Universal**: `fetch([opcode_x, opcode_y, delta_stack, sts, is_write], query=5*cursor+1, key=position)` — attention head looks up what instruction is at the current cursor
- **Specialized**: `_cursor_lookup(opx_vals)` — piecewise-constant FFN: for each cursor value i, emit `values[i] - values[i-1]` via a ReGLU pair `reglu(1, cursor - i + 1) - reglu(1, cursor - i)`

**7. Stack Access**
- `stack_top_value`: fetch 4 bytes with `key = stack_depth`, reconstructed as 32-bit integer
- `stack_second_value`: second from top (for binary ops like add/sub)
- `stack_third_position`: for `select` instruction
- `LOCAL_STRIDE = 256` bytes per call frame for local variable addressing

**8. Memory System**
- Content-addressable: each byte store writes `(address, value)` pair; each load fetches by address
- `memory_read_address = stack_top_value + immediate + byte_index`
- Most recent write wins (hardmax attention with latest tiebreak)
- `clear_key` mechanism invalidates stale writes when address changes

**9. Byte-Level Arithmetic**
- All 32-bit values encoded as 4 bytes (0–255), processed byte by byte
- `add_byte = (second_byte + top_byte + carry) mod 256`
- `sub_byte = (second_byte - top_byte - borrow) mod 256`
- Carry/borrow: `carry_out = stepglu(one, second_byte + top_byte + carry_in - 256)`
- `stepglu` acts as a threshold — exactly 1 when sum ≥ 256, else 0

**10. Comparisons**
- Unsigned: `a_gt_b_u = stepglu(one, second - top - 1)`
- Signed: adjusts for MSB difference (sign bit at position 31)
- `cond_nonzero = stepglu(one, top_value - 1)` — 1 when top-of-stack ≠ 0

**11. Call Stack**
- Return addresses stored with `key = call_depth * 4 + byte_index`
- Depth tracked via `delta_call_depth` (+1 on call, -1 on return)

**12. Result Byte — Combining All Operations**
Combines results of all instruction types:
- `i32.const` → constant byte from immediate
- `i32.add` → add_byte
- `i32.sub` → sub_byte  
- memory loads → memory_byte (with sign extension for `_s` variants)
- comparisons → 0 or 1 in byte 0
- branches → branch offset bytes

Each is gated by `is_op()` so only the matching opcode contributes.

**13. Next-Token Prediction**
Output token scores use quadratic loss:
```
score("7F") = H * emit_flag + (2 * 0x7F) * result_byte - 0x7F²
            = H * emit_flag - (result_byte - 0x7F)²  + const
```
This is minimized when `result_byte = 0x7F`, so the argmax picks the correct byte token.

### `WASMMachine` Class

Wrapper that calls `reset_graph()`, calls `build(program)`, and returns a `ProgramGraph`.

---

## Compilation Pipeline — `compilation/`

### `runtime.h` — The Handmade C Standard Library

**Why it exists:** Normal C programs use `printf`, `scanf`, etc. which require an OS. This VM has no OS. So this file replaces the standard library with minimal versions that only use ADD and SUB — no multiply, no divide, because the transformer can't simulate those directly.

| Function | What it does | Key trick |
|----------|-------------|-----------|
| `putchar(ch)` | Imported WASM function — the only "system call" | Declared as WASM import |
| `str_len(s)` | String length via index counting | — |
| `print_str(s)` | Outputs string via putchar loop | — |
| `parse_int(p)` | Parses decimal integer | `n*10 = n*8 + n*2 = ((n+n)+(n+n)+(n+n)+(n+n)) + (n+n)` — no MUL |
| `print_int(n)` | Decimal output | Finds largest power-of-10 via subtraction loops — no DIV |
| `sscanf(str, fmt, ...)` | Minimal `%d`-only parser | — |
| `printf(fmt, ...)` | Basic `%d`, `%s`, `%c`, `%%` support | — |

### `decoder.py` — WASM Binary Parser

Full MVP WASM binary decoder. The `.wasm` file is raw bytes. This file reads those bytes and produces Python objects you can work with.

**LEB128 encoding:** WASM stores integers in variable-length format. `_read_unsigned_leb128` and `_read_signed_leb128` decode these. A byte with bit 7 set means "more bytes follow"; bit 7 clear means "last byte".

| Section | Handler | What it parses |
|---------|---------|----------------|
| Type (1) | `_decode_type_section` | Function type signatures (params, results) |
| Import (2) | `_decode_import_section` | Functions, tables, memory, globals from host |
| Function (3) | `_decode_function_section` | Type indices for each function |
| Export (7) | `_decode_export_section` | Exported names and kinds |
| Global (6) | `_decode_global_section` | Global variables with init expressions |
| Code (10) | `_decode_code_section` | Function bodies (locals + instructions) |
| Data (11) | `_decode_data_section` | Memory initialization segments |

Key data classes: `WasmInstr`, `FuncType`, `FuncBody`, `DataSegment`, `WasmModule`.

**Important:** This step does NOT change the semantics. It just makes the binary readable as Python objects.

### `lower.py` — Hard-Op Lowering Pass

**Why it exists:** The transformer was built to simulate only basic WASM ops. Instructions like `i32.mul`, `i32.shl`, `i32.and` are not in the transformer's repertoire. This file replaces them with sequences of basic ops.

**Trigger condition:** Only lowers when the preceding instruction is `i32.const C` (constant operand). Runtime (non-constant) operands use loop-based implementations.

| Operation | Lowering Strategy |
|-----------|-------------------|
| `MUL × C` | Binary addition chain: represent C in binary, then double-and-add. `x*13` = start with x, double→2x, add x→3x, double→6x, double→12x, add x→13x |
| `DIV_U / C` | Subtraction loop: `q=0; while x >= C: x -= C; q++` |
| `REM_U % C` | Subtraction loop returning remainder |
| `DIV_S / C` | Make both operands positive, do unsigned divide, negate if signs differed |
| `AND & 0xFF` | `STORE8` + `LOAD8_U` at scratch address 0 |
| `AND & C` | Break C into 4 bytes; for each byte, extract bits via threshold comparisons |
| `OR \| C` | Byte-by-byte bit operation via `_emit_byte_bitop` |
| `XOR ^ 0xFFFFFFFF` | Special case: `255 - byte` for each byte |
| `XOR ^ C` | Byte-by-byte via `_emit_byte_bitop` |
| `SHL << C` | Byte-shift via memory: store at `SCRATCH+q`, load from `SCRATCH` (shifts by 8*q bits), then double `r` times |
| `SHR_U >> C` | Memory byte-shift in reverse + unsigned divide for remainder bits |
| `CLZ` | Loop: double x until bit 31 is set, count iterations |
| `CTZ` | Loop: extract low bit (store8/load8 + mod-2), check if 1, else right-shift by 1 (div by 2) |
| `POPCNT` | Loop: extract low bit, add to count, right-shift until zero |
| `ROTL/ROTR by C` | `rotl(x, c) = (x << c) + (x >> (32-c))` — uses SHL and SHR expansions above |
| `EXTEND8_S` | store8/load8_u + check if ≥ 128, if so subtract 256 |
| `EXTEND16_S` | store16/load16_s at scratch address |

`SCRATCH_ADDR = 0` (address 0 in linear memory) is the scratch pad used for byte extraction.

### `compile_wasm.py` — The Full Pipeline

| Function | What it does |
|----------|-------------|
| `find_clang()` | Locates clang with wasm32 target support (env var → PATH → fallback locations) |
| `compile_c_to_wasm(c_path)` | `clang --target=wasm32 -nostdlib -O2 -fno-builtin -fno-jump-tables` with `runtime.h` auto-included |
| `compile_function(func, mod)` | Compiles one function to flat dispatch table: resolves `block/loop/if/end` → `br/br_if` with numeric targets; handles `global.get/set` → memory ops |
| `build_program(mod)` | Full program: local init prologue → memory init → main body → all called functions → resolve call targets → convert absolute to relative offsets |
| `format_prefix(program)` | Converts dispatch table to `{ op b0 b1 b2 b3 ... }` text format |
| `format_input_section(input_str)` | Formats input as char/hex tokens + null byte + commit token |
| `compile_wasm_to_prefix(wasm_path)` | Full pipeline: decode → lower → build → format |
| `compile_program(input_path, args_str)` | High-level: detects .c/.wasm, writes `.txt` and `_spec.txt` files |
| `ensure_data()` | Ensures all examples compiled + reference traces generated |

**Control flow translation:**
- `block` / `loop` → pushed to `label_stack`
- `if` → `i32.eqz` + `br_if PLACEHOLDER`
- `else` → patch if's jump, emit unconditional `br PLACEHOLDER`
- `end` → pop stack, patch all pending PLACEHOLDERs to current position
- `br N` inside loop → backward jump to loop start
- `br N` inside block → forward jump to block end (placeholder, patched at `end`)

**`GLOBAL_BASE = 8`**: globals stored at memory addresses 8, 12, 16, ...

**`return` encoding**: For helper functions, `return` is encoded as a relative offset `~(distance_from_func_start)` — negative value signals "return" to the runner.

---

## WASM Reference Executor — `wasm/reference.py`

**Purpose:** The answer key. A plain Python WASM interpreter — no transformer, no ML, just a normal program that directly executes the compiled WASM dispatch table. Used to:
1. Verify the transformer produces correct output (compare token-by-token)
2. Generate reference trace files (`_ref.txt`) for automated testing

**How it works:**
- Maintains real Python stack, locals, memory, call stack, PC
- Executes each instruction directly
- **Produces the exact same token sequence** the transformer should generate — not just the final output text, but every intermediate byte token and commit token

**Token trace format:** Each WASM instruction produces:
- 4 result byte tokens: `7f 00 00 00` (result value, byte by byte, with carry marks)
- 1 commit token: `commit(+1,sts=1,bt=0)` (stack delta, store-to-stack flag, branch-taken flag)
- For output instructions: `out(H)` token before the commit
- For branches: `branch_taken` token if the branch was taken

This format matches exactly what `wasm/interpreter.py` (the computation graph) is designed to produce. The reference executor IS the ground truth for what the transformer should output.

---

## MILP Scheduler — `scheduler/milp.py`

**Purpose:** Solve an optimization problem: given all the operations in the computation graph and their dependencies, assign each operation to a transformer layer to minimize the total number of layers (and thus minimize d_model).

**The puzzle:** Some operations depend on others — you can't compute B until A is done. Operations are assigned to layers (each layer has 4 phases: attn, persist1, ffn, persist2). The MILP solver finds the optimal assignment.

### Layer Structure

Each transformer layer has 4 phases:
```
Phase 0 (attn):    LookUp (attention) heads execute
Phase 1 (persist1): Results saved to slots via linear projection
Phase 2 (ffn):     ReGLU gates execute
Phase 3 (persist2): Results saved to slots via linear projection
```

### Key Functions

| Function | What it does |
|----------|-------------|
| `_build_graph(all_dims, all_lookups)` | Builds dependency DAG: which operations must precede which |
| `_min_layers(ops, op_deps)` | Computes critical path (minimum possible layers, ASAP schedule) |
| `milp_schedule(...)` | **Main solver** — minimizes D_half (half of d_model) |
| `interval_coloring(all_dims, dim_birth, dim_death)` | Greedy slot assignment with min-heap reuse |
| `_write_plan(path, ...)` | Saves schedule to plan.yaml |

### MILP Formulation

**Decision variables:**
- `k[op]`: Layer assignment (0 to N-1) for each operation
- `z[op]`: Phase selector (persist1=0 or persist2=1 for PersistDimensions)

**Objective:** Minimize `D_half` = max pathwidth (number of simultaneously-alive dimensions) across all layer boundaries

**Constraints:**
- Data dependencies: child must be in a later phase than parent
- Tight constraints: ReGLUs/persists that consume average-tiebreak LookUps must be in the same layer
- FFN width: at most `max_ffn` ReGLUs per layer
- Width: occupied slots + head requirements cannot exceed D_half at any boundary

**Solver:** HiGHS (preferred) or CBC via PuLP library, 1-hour time limit.

**Output:** A `plan.yaml` file listing which operations go in which layer/phase. `weights.py` reads this plan to construct the model.

---

## Model & Weights — `model/`

### `transformer.py` — The Empty Machine

Defines the transformer architecture. No logic for what the weights mean — just the structure.

```python
class VanillaTransformer(nn.Module):
    tok     = Embedding(vocab, d_model)            # token → d_model vector
    attn    = [MultiheadAttention(...)] * n_layers # attention layers
    ff_in   = [Linear(d_model, 2*d_ffn)] * n_layers # FFN input (gate + value halves)
    ff_out  = [Linear(d_ffn, d_model)] * n_layers  # FFN output
    head    = Linear(d_model, vocab)               # residual → token scores
```

**Position encoding** — injected into slots 0, 1, 2 before any layer runs:
```python
def add_position_encoding(x, pos):
    x[0] += pos                                     # slot 0 = position
    x[1] += 1.0/math.log(2) - 1.0/math.log(pos+2) # slot 1 = inv_log_pos
    x[2] += pos * pos                               # slot 2 = position_sq
```

**FFN uses ReGLU:**
```python
gate, val = ff_in(x).chunk(2)          # split into gate half + value half
x = x + ff_out(F.relu(gate) * val)    # relu(gate) * val — the gated activation
```
This is the physical realization of `ReGLUDimension(a_expr, b_expr)`:
- `b_expr` → baked into gate row of `ff_in`
- `a_expr` → baked into value row of `ff_in`
- `relu(gate) * val` = `relu(b) * a`

**`generate_with_cache()` — the inference loop:**
1. For each position in the prefix: look up embedding, add position encoding, run all layers (updating KV cache)
2. After the last prefix token: generate new tokens one per forward pass via `head(x).argmax()`
3. Each new token is appended and processed in the next iteration
4. Stop when `halt` token is generated

**`HARD_K = 1e10`** is multiplied into the Q vectors when constructing weights. With very large Q, softmax(Q·K / sqrt(d)) becomes argmax — the highest-scoring key gets weight ~1.0 and all others get ~0. This converts softmax attention into hard (argmax) attention.

### `weights.py` — The Factory

Reads the blueprint (computation graph) + schedule (plan.yaml) and fills in every weight matrix with exact numbers.

**Step 1 — Slot assignment (register allocation):**
```python
FIXED = {position: 0, inv_log_pos: 1, position_sq: 2}  # always in slots 0,1,2
slot_of = dict(FIXED)
next_slot = 3
# assign one slot per Dimension, in order
```

Slot reuse: when a Dimension is "dead" (no future layers read it), its slot is freed and reused by a newly-born Dimension. This keeps d_model small. Implemented as interval coloring.

**The erase mechanism:** When a slot is reused, the old value must be zeroed first. A special "passthrough head" writes `-1 * current_value` back to the slot, netting to zero. This is `use_erase=True`.

**Step 2 — `expr_to_tensor(expr)`:**
```python
def expr_to_tensor(expr):
    w = torch.zeros(D)
    for dim, coeff in expr.terms.items():
        w[slot_of[dim]] += coeff
    return w
```
Converts a symbolic `Expression` like `2*stack_ptr + cursor` into a weight vector `[0, 0, 0, 2.0, 0, 1.0, ...]`.

**Step 3 — Embedding table:**
Each token's embedding row = `expr_to_tensor(input_tokens[token_name])`, with slots 0/1/2 cleared (position is added separately at runtime).

**Step 4 — Attention weights (per layer):**
For each `LookUp`:
```python
ip[h*2]        = expr_to_tensor(lu.query_exprs_2d[0]) * HARD_K  # Qx
ip[h*2 + 1]    = expr_to_tensor(lu.query_exprs_2d[1]) * HARD_K  # Qy
ip[D + h*2]    = expr_to_tensor(lu.key_exprs_2d[0])              # Kx
ip[D + h*2+1]  = expr_to_tensor(lu.key_exprs_2d[1])             # Ky
ip[2*D + h*2]  = expr_to_tensor(lu.value_exprs[0])              # V0
ip[2*D + h*2+1]= expr_to_tensor(lu.value_exprs[1])              # V1
op_w[slot_of[d0], h*2] = 1.0  # head output → slot
```

**Passthrough heads:** Some values just need to be copied forward without transformation. A passthrough head has Q=K=current position (always attends to itself), and V=source slot. These are packed 2 per head.

**Step 5 — FFN weights (per layer):**
For each `ReGLUDimension`:
```python
fi[j]          = expr_to_tensor(rg.b_expr)   # gate row (condition)
fi[d_ffn + j]  = expr_to_tensor(rg.a_expr)   # value row (content)
fo[slot_of[rg], j] = 1.0                      # output to slot
```

**Step 6 — Output head:**
```python
model.head.weight[tok_idx] = expr_to_tensor(output_tokens[tok_name])
```
Each output token's scoring row = the expression that evaluates high for that token.

### `save_weights` / `load_weights` — Binary File Format

```
[6 int32: vocab, d_model, n_layers, n_heads, d_ffn, stop_token_id]
[for each token: int32 length + UTF-8 bytes]
[embedding matrix: vocab × d_model float64]
[for each layer:
    in_proj_weight: 3*d_model × d_model float64
    out_proj.weight: d_model × d_model float64
    ff_in.weight: 2*d_ffn × d_model float64
    ff_out.weight: d_model × d_ffn float64]
[output head: vocab × d_model float64]
[has_erase flag int32]
[if has_erase: per-layer lists of erased slots]
[has_tiebreak flag int32]
[if has_tiebreak: per-layer per-head tiebreak flags]
```

This binary file is read directly by the C++ inference engine.

---

## Attention System — `attention/`

### The Two Caches

| Cache | Algorithm | Complexity | Use |
|-------|-----------|-----------|-----|
| `HullKVCache` | 2D convex hull trick | O(log n) per query | Default — needed for long programs |
| `StandardKVCache` | Standard softmax | O(n) per query | Debugging (`--nohull` flag) |

For Sudoku generating ~900K tokens, O(n) vs O(log n) is the difference between hours and seconds.

### `hull2d_cht.h` — O(log n) Hard Attention

**Key insight:** Maximizing `q · k = qx*kx + qy*ky` with `qy ≠ 0` is equivalent to maximizing `kx * (qx/qy) + ky` — a 1D "which line is highest at position m?" query, answered by the convex hull trick.

| Struct | Purpose |
|--------|---------|
| `HullMeta` | Aggregates tied values (vsum, vlast, count) for tie-breaking (average or latest) |
| `_HullCHT` | Dynamic convex hull trick: stores lines sorted by slope; `argmax(x)` = binary search on breakpoints |
| `HullHalf` | Wrapper: `is_upper=true` maintains max envelope; `is_upper=false` maintains min (via negation) |
| `HardAttentionHead` | Full 2D: upper hull (qy>0) + lower hull (qy<0) + edge cases for qy=0 |
| `BruteAttentionHead` | O(n) linear scan reference for testing correctness |

**`_HullCHT` internals:**
- Lines stored in `std::multiset` sorted by slope `m`
- Each line stores `p` = the x-coordinate where it stops being optimal
- `add_line(m, b)`: insert new line, remove redundant lines whose "best" interval is now 0-width
- `argmax(x)`: `lower_bound(x)` on the `p` values → O(log n)

**`HullHalf.query`:**
- Compute `m = qx/qy`
- Find best line via `cht.argmax(m)`
- Scan neighbors for ties (same dot-product score)
- Resolve ties via `HullMeta` (average or latest)

**`HardAttentionHead` edge cases:**
- `qy = 0, qx > 0` → want max `kx` → use `right_meta`
- `qy = 0, qx < 0` → want min `kx` → use `left_meta`
- `qy = 0, qx = 0` → all scores are 0 → return `global` (average of everything)

### `hull_ext.cpp` — pybind11 Bindings

Wraps `HardAttentionHead` into a Python-callable `HullKVCache` class:
```python
cache = HullKVCache(n_layers, n_heads)
cache.layer_step(layer, keys, queries, values, seq)  # → (n_heads, 2) numpy array
cache.set_tiebreak(layer, head, latest=True/False)
cache.clear()
```

Keys, queries, values are passed as `(n_heads, 2)` numpy arrays.

### `standard_cache.py` — Softmax Reference

```python
scores = torch.einsum("thi,hi->th", K, Q)   # dot products with all past keys
weights = F.softmax(scores, dim=0)           # softmax over time
out = torch.einsum("th,thi->hi", weights, V) # weighted sum of values
```
O(n) — grows linearly with sequence length.

---

## CLI Entry Points

| Command | Entry Point | Purpose |
|---------|-------------|---------|
| `wasm-compile` | `compilation/compile_wasm.py:main` | C/WASM → token prefix `.txt` files |
| `wasm-build` | `build.py:main` | Build universal transformer weights (run once) |
| `wasm-run` | `runner.py:main` | Run programs through transformer inference |
| `wasm-eval` | `evaluator.py:main` | Exact graph evaluation (no weights, symbolic) |
| `wasm-specialize` | `specialize.py:main` | First Futamura Projection — bake program into weights |
| `wasm-reference` | `wasm/reference.py:main` | Generate ground-truth token traces |

### `build.py` — One-Time Setup

Runs once. Calls `wasm/interpreter.py` → `scheduler/milp.py` → `model/weights.py` and saves `model.bin`. After this, `model.bin` is used for all runs. Only needs to be re-run if the VM design changes.

### `runner.py` — Run Programs

Three steps in sequence:
1. Compile any missing C examples (`ensure_data()`)
2. Build model if `model.bin` doesn't exist (calls `build.py`)
3. Run each `.txt` file through the model (C++ engine by default, Python with `--python`)

Reports: PASS/FAIL (vs reference), tokens/sec, wasm-ops/sec, FLOPs/token.

**Two backends:**
- **C++ engine** (default): Compiles `transformer.cpp` with BLAS, runs with hull cache
- **Python fallback** (`--python`): Uses PyTorch `generate_with_cache()`

Flags: `--nohull` (use `StandardKVCache`), `--verbose` (print full token sequence), `--max-new-tokens N`

### `evaluator.py` — Exact Graph Evaluation

Runs the computation graph directly with exact arithmetic — no weight matrices, no PyTorch. Each dimension's value is computed symbolically by evaluating the Expression formulas. Used for debugging: if the graph evaluator gives wrong answers, the bug is in the graph; if the graph evaluator is right but the model is wrong, the bug is in weight construction.

### `specialize.py` — First Futamura Projection

```python
instructions = parse_program("collatz.txt")   # read compiled program
pg = WASMMachine(program=instructions).build() # build specialized graph
model = build_model(program_graph=pg)          # construct specialized weights
```

1. Parse the `{ ... }` block from the `.txt` file into a list of instructions
2. Build a `WASMMachine` with those instructions — this bakes the program's opcodes into `_cursor_lookup()` piecewise-constant FFN patterns (no attention needed to fetch instructions)
3. Build and save a smaller, faster model that takes only `start + input + commit` tokens

**Result:** The specialized model's input is just `start World commit(+0,sts=0,bt=0)` instead of the full `{ ... } World commit(...)`. The program's logic lives in the FFN weights.

### `wasm/reference.py` — Answer Key Generator

Directly executes compiled programs and records every token the transformer should have generated. Used in two ways:
- `generate_all()` called by `ensure_data()` — creates `_ref.txt` files for all examples
- `runner.py` compares model output token-by-token against `_ref.txt`

---

## Example Programs

Located in `transformer_vm/examples/`. All use `compute(const char *input)` from `runtime.h`.

| Program | Description | Notable |
|---------|-------------|---------|
| `hello.c` | Simple string output | Minimal test case, no input parsing |
| `collatz.c` | Collatz sequence | Only ADD/SUB; uses sscanf + printf from runtime |
| `fibonacci.c` | Fibonacci numbers | Uses sscanf/printf |
| `addition.c` | Long addition with carry | Tests carry propagation |
| `min_cost_matching.c` | Hungarian algorithm | Complex control flow |
| `sudoku.c` | Constraint propagation solver | ~900K tokens generated, ~30K tok/sec |
| `lowering_test.c` | Tests for lowered operations | Validates MUL/DIV/bitwise lowering |

---

## Key Concepts & Insights

### 1. No Training Required
All transformer weights are derived **algebraically** from the computation graph. The MILP solver minimizes architecture size (d_model, n_layers), and `weights.py` fills in exact values using `expr_to_tensor()`. Zero gradient descent.

### 2. The Residual Stream as Named Variables
Each slot in the d_model vector has a specific, named meaning (e.g., slot 3 = stack_depth, slot 7 = cursor). The graph DSL lets you name and describe relationships between slots. `weights.py` then hard-codes those relationships into weight matrices.

### 3. Circle-Point Opcode Dispatch
35 opcodes → 35 unique points on circle `x² + y² = 32045`. A single ReGLU neuron `relu(px*opcode_x + py*opcode_y - 32045 + 1)` activates for exactly one opcode. Replaces a switch-case with pure linear algebra.

### 4. Byte-Level Arithmetic
All 32-bit integers encoded as 4 bytes (0–255). Carries/borrows propagated explicitly with `stepglu` threshold detection. Enables ADD/SUB with no multiply in the transformer.

### 5. Content-Addressable Memory
Stack, locals, and memory all use `fetch()` with address-based keys. Writing stores `(address, value)` pairs; reading queries by address. The hardmax attention retrieves the most recent write to each address. `clear_key` invalidates stale entries.

### 6. `HARD_K = 1e10` — Softmax Becomes Argmax
The Q vectors are multiplied by `1e10` when stored in `in_proj_weight`. This makes `softmax(Q·K / sqrt(d))` concentrate all weight on the single highest-scoring key, making the attention effectively argmax (hardmax) without changing the architecture.

### 7. O(log n) Hard Attention
The 2D key encoding (`kx=2k, ky=-k²`) turns "find best matching key" into "find the highest line at position m=qx/qy". The convex hull trick answers this in O(log n). Critical for programs generating hundreds of thousands of tokens.

### 8. First Futamura Projection
Specialization bakes program bytecode into FFN weights using piecewise-constant ReGLU pairs. The program no longer appears in the input — the transformer **is** the program. Reduces input length and model size.

### 9. Quadratic Token Scoring
Output tokens are scored with `H * emit_flag - (computed - target)²`. The quadratic loss is minimized at `computed = target`, ensuring the correct byte token wins argmax without any threshold tuning.

### 10. Hard-Op Lowering
MUL, DIV, bitwise ops are expanded at compile time. Example: `x * 7` via binary addition chain:
- 7 = 111₂
- Start: `acc = x`
- Bit 1 (value 2): double → `acc = 2x`, bit set → `acc = 2x + x = 3x`
- Bit 0 (value 1): double → `acc = 6x`, bit set → `acc = 6x + x = 7x`
Result: 4 ADD instructions instead of 6 naive additions (binary method is optimal).

### 11. Slot Reuse = Register Allocation
The scheduler tracks when each Dimension is "born" (first computed) and "dies" (last used). Dead slots are freed and reused by new Dimensions — exactly like a compiler assigning variables to CPU registers. This keeps d_model small.

### 12. The Erase Mechanism
When a slot is reused, its old value must be zeroed before the new value arrives. A passthrough attention head writes `-1 × old_value` back to the slot (net result: slot becomes 0). This is the "erase" in `use_erase=True`.

---

## Common Confusions Clarified

**Q: Is the C file sent to the model?**
No. The C file is compiled entirely on the CPU into a `.txt` file. The model never sees C or WASM.

**Q: Is the `.txt` file just a text version of the `.wasm` file?**
No. Three major transformations happen: (1) hard ops are expanded into loops of simple ops, (2) structured control flow is flattened to numeric jumps, (3) all functions are merged into one flat table. A single `i32.mul` instruction can become 30+ instructions in the `.txt`.

**Q: Is the model trained on programs?**
No. The weights are mathematically computed from the formal description of what a WASM VM needs to compute. The MILP solver minimizes the architecture size; `weights.py` fills in exact values. No gradient descent ever runs.

**Q: Why is d_model so small (36)?**
Because the weights are exact — every slot has a precise meaning. Normal trained models need many neurons to approximately represent information; here each neuron does a specific job determined by the graph DSL.

**Q: Why use WASM instead of compiling C directly to the model's instruction set?**
WASM is a clean, well-defined stack machine with simple structured control flow. It's much easier to parse and work with than x86. `clang` already compiles C to WASM efficiently.

**Q: What is `commit(+1,sts=1,bt=0)`?**
A token that marks the end of one WASM instruction's execution. The values encode: stack delta (+1 means stack grows by 1), store-to-stack flag (1 means the result is written to stack), branch-taken flag (0 means no branch). The model generates one commit token per instruction executed.

---

## Codebase Statistics

| Component | Language | Lines |
|-----------|----------|-------|
| Core DSL (`graph/`) | Python | ~450 |
| WASM Interpreter (`wasm/interpreter.py`) | Python | ~640 |
| WASM Reference (`wasm/reference.py`) | Python | ~670 |
| Compilation (`compilation/`) | Python + C | ~2,600 |
| Scheduler (`scheduler/`) | Python | ~815 |
| Model (`model/`) | Python + C++ | ~1,350 |
| Attention (`attention/`) | C++ + Python | ~560 |
| CLI tools | Python | ~940 |
| Tests | Python | ~250 |
| **Total** | | **~9,100** |
