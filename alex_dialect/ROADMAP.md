# Alex Dialect: Learning Roadmap and Progress

This file tracks how far the `alex_dialect` project has gone against the MLIR
learning roadmap, and what to build next.

Final goal: contribute to Torch-MLIR, ONNX-MLIR or another MLIR-based project.

---

## 1. Where things stand

| # | Roadmap stage | Status | Evidence in this repo |
|---|---------------|--------|-----------------------|
| 1 | ML/DL fundamentals | 🟡 Partial | Tensors, shapes, broadcasting (`addBCLinalg`), reduction (`reductionLinalg`), PyTorch-style ops (`add`, `addcmul`) |
| 2 | Compiler fundamentals | 🟡 Partial | SSA, IR and lowering are used every day. Not yet studied: AST and front ends, optimization theory |
| 3 | LLVM basics | 🔴 Light | Lowered to the LLVM dialect (assignment 5), but LLVM IR itself hasn't been read or optimized |
| 4 | MLIR fundamentals | 🟢 Done | Dialect, ops, values, types, regions and blocks (built a `linalg.generic` body by hand) |
| 5 | MLIR development | 🟡 Partial | ✅ TableGen/ODS, type constraints, optional operands, conversion patterns. ❌ Verifiers, folding, canonicalization, assembly format, traits, interfaces, DRR/PDLL |
| 6 | MLIR lowering | 🟡 Partial | ✅ `alex` → `arith`, `alex` → `linalg` (`generic`, `elementwise`, broadcast maps, reduction), `tensor.empty`, `linalg.fill`. ❌ Running the full pipeline to LLVM and executing it, bufferization in depth, `affine`, `vector`, `TypeConverter` |
| 7 | MLIR testing | 🟡 Partial | ✅ lit + FileCheck for the `arith` lowering, failure tests, pytest for the Python bindings. ❌ `test/linalg/*.mlir` have no `RUN`/`CHECK` lines |
| 8 | Build MLIR projects | 🟡 In progress | This repo: custom ML ops, the `alex-opt` tool, a C API and nanobind Python bindings |
| 9 | Study existing projects | 🔴 Not started | |
| 10 | Open-source contributions | 🔴 Not started | |

### Project timeline (from git history)

1. Built the `alex-opt` tool and an empty dialect.
2. Assignment 1: a `torch.add`-like op with an optional operand.
3. Assignment 2: type constraints.
4. Assignment 3: `add`, `sub`, `mul` and `constant` ops.
5. Assignment 4: lowering to built-in dialects (`arith`), plus lit/FileCheck tests and failure cases.
6. Python bindings through the C API and nanobind, plus pytest.
7. Assignment 5: lowering to LLVM.
8. Assignment 6: an `addcmul` op with 1D, 2D, 3D and dynamic-shape tests.
9. Current branch, `try/accm_add_div`: lowering to `linalg` with `linalg.elementwise`, broadcasting and a `linalg.generic` reduction (zero-initialized accumulator, int/float, any rank).

---

## 2. Next steps, in order

Each step is a small, self-contained task in this repo. Do them on separate
branches, like the earlier assignments.

### Step 1: Finish the linalg branch properly (testing)
- [ ] Add `// RUN:` and `// CHECK:` lines to `test/linalg/linalg.mlir` and `test/linalg/reduction.mlir`.
- [ ] Add tests for an f32 reduction, a 2-D reduction and a broadcast add (`tensor<2x3xf32> + tensor<3xf32>`).
- [ ] Add a failure test, for example a reduction whose result type doesn't match the input element type.
- [ ] Fix the copy-pasted `summary`/`description` of `reductionLinalgOp` in `AlexOps.td`.

**Learn:** FileCheck directives (`CHECK-LABEL`, `CHECK-NEXT`, `CHECK-SAME`, `CHECK-NOT`, `[[VAR:%.*]]` captures) and `--split-input-file` with `expected-error`.

### Step 2: Run your IR end to end (lowering + LLVM basics)
- [ ] Lower `reduction.mlir` all the way to the LLVM dialect:
  ```bash
  alex-opt test/linalg/reduction.mlir \
    --convert-alex-to-linalg \
    --one-shot-bufferize="bufferize-function-boundaries" \
    --convert-linalg-to-loops --convert-scf-to-cf \
    --expand-strided-metadata --finalize-memref-to-llvm \
    --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm \
    --reconcile-unrealized-casts
  ```
- [ ] Translate it to LLVM IR with `mlir-translate --mlir-to-llvmir`, then **read the `.ll` file**.
- [ ] Execute it with `mlir-runner` or the Python `ExecutionEngine` and confirm it returns `60`.
- [ ] Run `opt -O2` on the `.ll` file and compare the result with the unoptimized version.
- [ ] Register this sequence as a named pipeline (for example `--alex-to-llvm`) with `PassPipelineRegistration`.

**Learn:** bufferization (tensor vs. memref, `one-shot-bufferize`), LLVM IR basics (basic blocks, `phi`, `load`/`store`, `getelementptr`).

### Step 3: Make your ops "real" MLIR ops (MLIR development)
- [ ] Add **verifiers** (`let hasVerifier = 1;`) so invalid IR is rejected at parse time instead of in the lowering.
- [ ] Add **traits**: `Pure`, `SameOperandsAndResultType`, `SameOperandsAndResultElementType`.
- [ ] Add an **`assemblyFormat`** so you can write `alex.add %a, %b : tensor<4xf32>`.
- [ ] Add a **folder** (`hasFolder`): for example, `alex.add` of two constants folds to a constant, and `x + 0` folds to `x`.
- [ ] Add a **canonicalization pattern** (`hasCanonicalizer`) and test it with `--canonicalize`.
- [ ] Rewrite one pattern with **DRR** (`Pat<>` in TableGen) or **PDLL**.

**Learn:** the greedy pattern driver vs. dialect conversion; how `fold` differs from `canonicalize`.

### Step 4: Interfaces and types
- [ ] Implement `InferTypeOpInterface` so result types are inferred.
- [ ] Define a custom type (`TypeDef`) and lower it with a **`TypeConverter`**.
- [ ] Move the passes to TableGen (`Passes.td`) instead of hand-written `PassWrapper` classes.

### Step 5: Linalg transformations (why lowering to linalg pays off)
- [ ] Generalize: `--linalg-generalize-named-ops` (`elementwise` → `generic`).
- [ ] Tile, fuse and vectorize your ops with the **Transform dialect**.
- [ ] Try an `affine` lowering path (`--convert-linalg-to-affine-loops`) and the affine optimizations.

### Step 6: A real ML op end to end (project milestone)
Implement one op that real models need, through every layer of this repo: ODS
op, verifier, linalg lowering, FileCheck tests, a Python binding and execution
to the correct numbers. Good candidates:
- `softmax` (reduction + broadcast + elementwise: everything combined)
- `matmul` (lower to `linalg.matmul`, then tile it)
- `relu` / `gelu`, or `layernorm`

---

## 3. Moving toward open source

### Study existing projects (stage 9)
- **Toy tutorial**, chapters 1–7 (mlir.llvm.org/docs/Tutorials/Toy). Compare each chapter with what this repo already does.
- **Upstream `TosaToLinalg`** (`mlir/lib/Conversion/TosaToLinalg`), the closest real-world version of `AlexToLinalg.cpp`.
- **Torch-MLIR**: read how one op flows `torch` → `linalg`
  (`lib/Conversion/TorchToLinalg/`). Pick an op you've already implemented (`add`, `addcmul`) and compare.
- **ONNX-MLIR**: the `onnx` → `krnl` → `affine` lowering.

For each project, write down the dialects, the pass pipeline, how tests are
organized and how to build and run one test.

### First contributions (stage 10)
1. Build Torch-MLIR from source and run its test suite.
2. Read `CONTRIBUTING.md`, the code review process and the CI setup.
3. Look for issues labelled `good first issue` / `help wanted`.
4. Start small: fix docs, add missing tests, improve error messages.
5. Then op support: Torch-MLIR regularly needs new `aten` op lowerings. That's the same work as `addcmul` in this repo.

---

## 4. Reference sheet

| Topic | Where |
|-------|-------|
| MLIR docs | https://mlir.llvm.org/docs/ |
| ODS reference | https://mlir.llvm.org/docs/DefiningDialects/Operations/ |
| Dialect conversion | https://mlir.llvm.org/docs/DialectConversion/ |
| Bufferization | https://mlir.llvm.org/docs/Bufferization/ |
| Linalg rationale | https://mlir.llvm.org/docs/Dialects/Linalg/ |
| Transform dialect | https://mlir.llvm.org/docs/Dialects/Transform/ |
| FileCheck | https://llvm.org/docs/CommandGuide/FileCheck.html |
| LLVM language reference | https://llvm.org/docs/LangRef.html |
| Jeremy Kun, "MLIR for the impatient" | https://www.jeremykun.com/ |
| Torch-MLIR | https://github.com/llvm/torch-mlir |
| ONNX-MLIR | https://github.com/onnx/onnx-mlir |

### Useful debugging flags
```bash
--mlir-print-ir-after-all          # IR after every pass
--mlir-print-op-generic            # generic op syntax
--debug-only=dialect-conversion    # why a conversion failed (needs a debug build)
--verify-each                      # run the verifier after every pass
```
