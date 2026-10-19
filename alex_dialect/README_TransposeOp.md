# `alex.transpose`: verifier, builder, folder, canonicalizer

This note walks through five steps that turn `alex.transpose` from "an op that
can be lowered" into "an op that behaves like a real MLIR op":

| Step | What | Roadmap topic |
|---|---|---|
| 1 | Shape helper | Development |
| 2 | Verifier + negative test | Development, Testing |
| 3 | Custom builder | Development (ODS) |
| 4 | Folder + canonicalization pattern + `Pure` | Development (folding, canonicalization) |
| 5 | Simplify the lowering | Lowering |

Every snippet here was built and tested against this repo. When you are done,
these should all pass:

```bash
lit -v test/transpose
```

Files you will touch:

```
include/Alex/AlexOps.td        steps 2, 3, 4  (declare: hasVerifier, builders, hasFolder, ...)
include/Alex/AlexOps.h         steps 1, 4     (helper declarations, Pure header)
lib/AlexOps.cpp                steps 1-4      (all the C++ bodies)
lib/Conversion/AlexToArith/AlexToLinalg.cpp   step 5
test/transpose/transpose_invalid.mlir         step 2 (new)
test/transpose/transpose_canonicalize.mlir    step 4 (new)
```

---

## The big idea: ODS declares, C++ defines

Everything below follows one rule:

- In **`.td`** you *declare* that the op has a feature:
  `let hasVerifier = 1;`
- TableGen then *declares* a C++ method on the op class (in `AlexOps.h.inc`):
  `mlir::LogicalResult verify();`
- **You** write the body of that method in **`AlexOps.cpp`**.

If you set the flag but forget the body, you get a **linker** error
("undefined reference to `alex::transposeOp::verify()`"). If you write the body
but forget the flag, you get a **compiler** error (no such member). Both
messages mean the same thing: the `.td` and the `.cpp` don't agree.

---

## Step 1: Shape helper

### Why

Three places need to answer the same question:

> Given the input type and `dim0`/`dim1`, what is the result type?

- the **verifier** (step 2) needs it to *check* the result type
- the **builder** (step 3) needs it to *compute* the result type
- the **lowering** (step 5) needs the normalized dims

Write it **once**, otherwise three copies drift apart.

### Where, and why not `AlexToLinalg.cpp`

It goes in `AlexOps.cpp`, next to the op. The reason is the **dependency
direction**:

```
MLIRAlex (ops)  <---  MLIRAlexToArith (conversion)
```

The conversion library depends on the ops library, never the other way round.
The verifier and builder live in the ops library, so they can't call anything
in `AlexToLinalg.cpp`. Helpers about *the op itself* belong with the op, and
the lowering is then free to use them too.

### Code

**`include/Alex/AlexOps.h`**: declare the helpers after the `.h.inc` include,
so they can be used from other files:

```cpp
#define GET_OP_CLASSES
#include "Alex/AlexOps.h.inc"

namespace alex {

/// Turns a torch-style dim (negative counts from the end) into 0..rank-1.
/// Fails if the dim is out of range.
mlir::FailureOr<int64_t> normalizeDim(int64_t dim, int64_t rank);

/// Result type of swapping dim0 and dim1 of inputType.
/// Fails if either dim is out of range.
mlir::FailureOr<mlir::RankedTensorType>
inferTransposeType(mlir::RankedTensorType inputType, int64_t dim0,
                   int64_t dim1);

} // namespace alex
```

**`lib/AlexOps.cpp`**: define them at the bottom of the file:

```cpp
mlir::FailureOr<int64_t> alex::normalizeDim(int64_t dim, int64_t rank) {
  if (dim < 0)
    dim += rank;
  if (dim < 0 || dim >= rank)
    return mlir::failure();
  return dim;
}

mlir::FailureOr<mlir::RankedTensorType>
alex::inferTransposeType(mlir::RankedTensorType inputType, int64_t dim0,
                         int64_t dim1) {
  int64_t rank = inputType.getRank();
  mlir::FailureOr<int64_t> d0 = normalizeDim(dim0, rank);
  mlir::FailureOr<int64_t> d1 = normalizeDim(dim1, rank);
  if (mlir::failed(d0) || mlir::failed(d1))
    return mlir::failure();

  llvm::SmallVector<int64_t> shape(inputType.getShape());
  std::swap(shape[*d0], shape[*d1]);
  return mlir::RankedTensorType::get(shape, inputType.getElementType());
}
```

### Things to notice

- **`FailureOr<T>`** is MLIR's "either a `T` or a failure". You check it with
  `mlir::failed(x)` and get the value with `*x`. It is the MLIR way to say
  "this can fail" without exceptions or magic values like `-1`. (Note that
  `-1` would be a terrible "error" value here, since it is a *valid* dim.)
- `shape` is a **copy** (`SmallVector`) of the input shape. `getShape()`
  returns an `ArrayRef`, which is a read-only view, so you can't swap in it
  directly.
- Nothing here is transpose-lowering-specific: no Linalg, no rewriter. It is
  pure "type math".

---

## Step 2: Verifier

### Why

Today, an invalid transpose is **accepted** by the parser:

```mlir
// 2x3x4 with dim0=0, dim1=2 should be 4x3x2, not 3x2x4
%0 = "alex.transpose"(%x) {dim0 = 0 : i32, dim1 = 2 : i32}
    : (tensor<2x3x4xf32>) -> tensor<3x2x4xf32>
```

It only fails much later, in the lowering, with a vague message:

```
error: failed to legalize operation 'alex.transpose' that was explicitly marked illegal
```

That error means "no pattern wanted it". It doesn't say what is *wrong*. A
verifier rejects bad IR **the moment it is parsed or created**, with a precise
message:

```
error: 'alex.transpose' op expected result type 'tensor<4x3x2xf32>', got 'tensor<3x2x4xf32>'
```

MLIR runs verifiers after parsing and after every pass, so every pattern,
folder and lowering can **assume the op is valid**. That's what makes step 5
possible.

### Code

**`include/Alex/AlexOps.td`**, inside `def transposeOp`:

```tablegen
    let hasVerifier = 1;
```

**`lib/AlexOps.cpp`**:

```cpp
mlir::LogicalResult alex::transposeOp::verify() {
  mlir::RankedTensorType inputType =
      llvm::dyn_cast<mlir::RankedTensorType>(getInput().getType());
  mlir::RankedTensorType resultType =
      llvm::dyn_cast<mlir::RankedTensorType>(getResult().getType());

  // Nothing to check against if a shape is unknown.
  if (!inputType || !resultType)
    return mlir::success();

  int64_t dim0 = getDim0Attr().getInt();
  int64_t dim1 = getDim1Attr().getInt();

  mlir::FailureOr<mlir::RankedTensorType> expected =
      inferTransposeType(inputType, dim0, dim1);
  if (mlir::failed(expected))
    return emitOpError() << "dims (" << dim0 << ", " << dim1
                         << ") out of range for rank " << inputType.getRank();

  if (resultType != *expected)
    return emitOpError() << "expected result type " << *expected << ", got "
                         << resultType;

  return mlir::success();
}
```

### Things to notice

- **`getDim0Attr().getInt()`, not `getDim0()`.** `I32Attr` is *signless*, and
  its generated getter `getDim0()` returns `uint32_t`, which would turn `-1`
  into `4294967295`. `IntegerAttr::getInt()` sign-extends, so `-1` stays `-1`.
  You hit this bug in the lowering already. Use the same fix everywhere.
- **`emitOpError()`** prefixes the message with `'alex.transpose' op`, and you
  can stream types and numbers into it with `<<`. It returns a failure, so
  `return emitOpError() << ...;` reports and fails in one line.
- **Unranked tensors**: `IntOrFloatTensor` also allows `tensor<*xf32>`. There is
  nothing to compare, so the verifier accepts them. (The lowering still
  rejects them; that's fine.)
- **Comparing types**: `resultType != *expected` works because MLIR types are
  *uniqued*. Two `tensor<4x3x2xf32>` are the same object, so `==` is a
  pointer compare.
- `dim0 == dim1` is **valid** (torch allows it; it's a no-op). Step 4 folds it away.

### Negative test

A negative test checks that bad IR is *rejected with the right message*. This
is the kind of test that reviewers in Torch-MLIR ask for.

**`test/transpose/transpose_invalid.mlir`** (new):

```mlir
// RUN: alex-opt %s -split-input-file -verify-diagnostics

func.func @dim_out_of_range(%arg0: tensor<2x3xf32>) -> tensor<3x2xf32> {
  // expected-error @+1 {{dims (0, 2) out of range for rank 2}}
  %0 = "alex.transpose"(%arg0) {dim0 = 0 : i32, dim1 = 2 : i32}
      : (tensor<2x3xf32>) -> tensor<3x2xf32>
  return %0 : tensor<3x2xf32>
}

// -----

func.func @negative_dim_out_of_range(%arg0: tensor<2x3xf32>) -> tensor<3x2xf32> {
  // expected-error @+1 {{dims (-3, 0) out of range for rank 2}}
  %0 = "alex.transpose"(%arg0) {dim0 = -3 : i32, dim1 = 0 : i32}
      : (tensor<2x3xf32>) -> tensor<3x2xf32>
  return %0 : tensor<3x2xf32>
}

// -----

func.func @wrong_result_shape(%arg0: tensor<2x3x4xf32>) -> tensor<3x2x4xf32> {
  // expected-error @+1 {{expected result type 'tensor<4x3x2xf32>', got 'tensor<3x2x4xf32>'}}
  %0 = "alex.transpose"(%arg0) {dim0 = 0 : i32, dim1 = 2 : i32}
      : (tensor<2x3x4xf32>) -> tensor<3x2x4xf32>
  return %0 : tensor<3x2x4xf32>
}
```

How it works:

- **`-verify-diagnostics`**: instead of printing errors, `alex-opt` compares
  them against the `expected-error` comments. The test passes only if every
  expected error appears **and** no unexpected error appears.
- **`@+1`** means "on the next line". The op starts on the line after the comment.
- **`{{...}}`** holds a substring of the message. It doesn't need to be the
  whole message.
- **`-split-input-file` with `// -----`** treats each chunk as a separate file.
  Without it, the first error stops parsing and cases 2 and 3 would never be
  checked.
- There is **no FileCheck** in the RUN line. `-verify-diagnostics` does the
  checking itself.

Try breaking it: change `rank 2` to `rank 3` in one comment and watch the test fail.

---

## Step 3: Custom builder

### Why

A **builder** is the C++ function that creates an op. ODS generates some
automatically. For `alex.transpose`, the generated ones need the result type
from the caller:

```cpp
// generated: caller must work out 4x3x2 by hand
alex::transposeOp::create(rewriter, loc, resultType, input,
                          rewriter.getI32IntegerAttr(0),
                          rewriter.getI32IntegerAttr(2));
```

You only use builders when **C++ creates ops**: in rewrite patterns, in
folders, and in frontends (a future `TorchToAlex` would create `alex.*` ops
everywhere). Making every caller compute the result type is repetitive and
easy to get wrong. A custom builder does it once, using step 1:

```cpp
// custom: the op computes its own result type
alex::transposeOp::create(rewriter, loc, input, /*dim0=*/0, /*dim1=*/2);
```

### Code

**`include/Alex/AlexOps.td`**, inside `def transposeOp`:

```tablegen
    let builders = [
        OpBuilder<(ins "mlir::Value":$input, "int32_t":$dim0, "int32_t":$dim1)>
    ];
```

This makes TableGen declare
`static void build(OpBuilder &, OperationState &, Value input, int32_t dim0, int32_t dim1);`
plus a matching `create(...)`. The `ins` list here is the **C++ parameter
list**. It has nothing to do with the op's `arguments`.

**`lib/AlexOps.cpp`**:

```cpp
void alex::transposeOp::build(mlir::OpBuilder &builder,
                              mlir::OperationState &state, mlir::Value input,
                              int32_t dim0, int32_t dim1) {
  mlir::RankedTensorType inputType =
      llvm::cast<mlir::RankedTensorType>(input.getType());
  mlir::FailureOr<mlir::RankedTensorType> resultType =
      inferTransposeType(inputType, dim0, dim1);
  assert(mlir::succeeded(resultType) && "transpose dims out of range");

  // Forward to the builder ODS generated (result type first).
  build(builder, state, *resultType, input, builder.getI32IntegerAttr(dim0),
        builder.getI32IntegerAttr(dim1));
}
```

### Things to notice

- **A custom builder usually forwards to a generated one.** You don't fill
  `OperationState` by hand. You compute the missing piece (the result type)
  and call the generated `build(...)`, which takes the result type first.
- **`llvm::cast`, not `dyn_cast`.** A builder has no way to return an error,
  so calling it with an unranked tensor is a *programmer* mistake, and `cast`
  asserts on it. Same idea for the `assert` on bad dims.
- **`getI32IntegerAttr(dim0)`** builds the attribute from a signed `int32_t`,
  so `-1` is stored correctly.
- After the builder runs, the **verifier also runs** on the new op (in debug
  builds and after each pass), so a builder bug can't silently produce bad IR.

### How it's tested

A builder is C++, so a `.mlir` file can't call it directly. Something that
creates ops has to use it. In this repo that's the canonicalization pattern in
step 4 (`NormalizeTransposeDims`), so the `@normalize_dims` test there is the
builder's test.

---

## Step 4: Folder, canonicalization pattern, `Pure`

### Why

These simplify the IR **before** lowering. Less IR in means less work and
better code out. MLIR has two tools for it, and they do different jobs:

| | **Folder** (`fold`) | **Canonicalization pattern** |
|---|---|---|
| Can return | an existing value, or a constant | anything |
| Can create new ops | **no** | yes |
| Runs in | `-canonicalize`, and also whenever a builder uses `createOrFold` | `-canonicalize` |
| Use for | "this op equals something that already exists" | "rewrite this op into a better form" |

For transpose:

- **Folder**: `transpose(x, d, d)` is just `x`, and
  `transpose(transpose(x, a, b), a, b)` is also `x`. Both answers already exist
  (`x`), so nothing is created. That's a fold.
- **Canonicalization pattern**: rewrite dims into one standard form
  (non-negative, `dim0 < dim1`). That creates a **new op** with different
  attributes, so it can't be a fold.

Why bother with a canonical form? `(1, 0)`, `(0, 1)`, `(-1, 0)` and `(0, -1)` all
mean the same thing on a 2-D tensor. If every transpose is written one way,
every later pattern only has to handle one form.

### Code: declarations

**`include/Alex/AlexOps.td`**, inside `def transposeOp`:

```tablegen
    let hasFolder = 1;
    let hasCanonicalizer = 1;
```

### Code: folder

**`lib/AlexOps.cpp`**:

```cpp
mlir::OpFoldResult alex::transposeOp::fold(FoldAdaptor adaptor) {
  mlir::RankedTensorType inputType =
      llvm::dyn_cast<mlir::RankedTensorType>(getInput().getType());
  if (!inputType)
    return {};
  int64_t rank = inputType.getRank();

  // The verifier already checked the dims, so these always succeed.
  int64_t dim0 = *normalizeDim(getDim0Attr().getInt(), rank);
  int64_t dim1 = *normalizeDim(getDim1Attr().getInt(), rank);

  // transpose(x, d, d) swaps a dim with itself: it is just x.
  if (dim0 == dim1)
    return getInput();

  // transpose(transpose(x, a, b), a, b) -> x  (also matches b, a).
  transposeOp inner = getInput().getDefiningOp<transposeOp>();
  if (!inner || inner.getInput().getType() != getType())
    return {};

  int64_t innerDim0 = *normalizeDim(inner.getDim0Attr().getInt(), rank);
  int64_t innerDim1 = *normalizeDim(inner.getDim1Attr().getInt(), rank);
  bool sameSwap = (innerDim0 == dim0 && innerDim1 == dim1) ||
                  (innerDim0 == dim1 && innerDim1 == dim0);
  if (!sameSwap)
    return {};

  return inner.getInput();
}
```

How to read a folder:

- **Return `{}`** means "can't fold, leave the op alone". This isn't an error.
- **Return a `Value`** means "replace my result with this value". The driver
  does the replacement, so you never call the rewriter.
- **`*normalizeDim(...)` with no check** is safe because the verifier (step 2)
  guarantees the dims are in range. That's the payoff of having a verifier.
- **`getDefiningOp<transposeOp>()`** asks "was my input produced by another
  transpose?" It returns null if not (block argument, other op, ...).
- **The type check** `inner.getInput().getType() != getType()` is a safety
  net: a fold must return a value of *exactly* the op's result type.
- `FoldAdaptor adaptor` gives the operands **as constant attributes** when
  they are constants. You'd use it for constant folding (e.g. `add(2, 3) -> 5`).
  We don't need it here.

### Code: canonicalization pattern

Also in **`lib/AlexOps.cpp`**. It needs one more include at the top of the file:

```cpp
#include "mlir/IR/PatternMatch.h"
```

(Without it you get `invalid use of incomplete type 'class mlir::RewritePatternSet'`.)

```cpp
namespace {

/// Rewrites dims into canonical form: non-negative and dim0 < dim1.
/// transpose(x, -1, 0) on rank 3 -> transpose(x, 0, 2)
struct NormalizeTransposeDims
    : public mlir::OpRewritePattern<alex::transposeOp> {
  using mlir::OpRewritePattern<alex::transposeOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(alex::transposeOp op,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::RankedTensorType inputType =
        llvm::dyn_cast<mlir::RankedTensorType>(op.getInput().getType());
    if (!inputType)
      return mlir::failure();
    int64_t rank = inputType.getRank();

    int64_t dim0 = op.getDim0Attr().getInt();
    int64_t dim1 = op.getDim1Attr().getInt();
    int64_t lo = std::min(*alex::normalizeDim(dim0, rank),
                          *alex::normalizeDim(dim1, rank));
    int64_t hi = std::max(*alex::normalizeDim(dim0, rank),
                          *alex::normalizeDim(dim1, rank));

    // Already canonical: report failure so the driver doesn't loop forever.
    if (dim0 == lo && dim1 == hi)
      return mlir::failure();

    // Uses the Step 3 builder: no result type needed.
    rewriter.replaceOpWithNewOp<alex::transposeOp>(op, op.getInput(), lo, hi);
    return mlir::success();
  }
};

} // namespace

void alex::transposeOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet &results, mlir::MLIRContext *context) {
  results.add<NormalizeTransposeDims>(context);
}
```

How to read it, compared with your lowering patterns:

- **`OpRewritePattern` + `PatternRewriter`**, not `OpConversionPattern` +
  `ConversionPatternRewriter`. Canonicalization isn't a dialect conversion:
  there's no "legal/illegal" target and no `OpAdaptor`.
- **The "already canonical" check is essential.** The canonicalizer keeps
  applying patterns until nothing changes. If this pattern returned `success()`
  on an op that's already `(0, 2)`, it would replace it with an identical op,
  forever. A pattern **must** fail when there's nothing to do.
- **`replaceOpWithNewOp<alex::transposeOp>(op, op.getInput(), lo, hi)`** creates
  the new op with the arguments after `op`, so it picks your **custom builder**
  from step 3 (Value, int32_t, int32_t). The result type is computed for you.
- The pattern lives in an **anonymous namespace**. It's an implementation
  detail of the op, and only `getCanonicalizationPatterns` needs to see it.

### `Pure`: why the dead transpose didn't disappear

With only the folder, this:

```mlir
%0 = "alex.transpose"(%arg0) {dim0 = 1 : i32, dim1 = 0 : i32} : ...
%1 = "alex.transpose"(%0)    {dim0 = 0 : i32, dim1 = -1 : i32} : ...
return %1
```

canonicalizes to:

```mlir
%0 = "alex.transpose"(%arg0) <{dim0 = 0 : i32, dim1 = 1 : i32}> : ...   // dead, but still here!
return %arg0
```

The fold worked (`return %arg0`), but `%0` is now unused and **still there**.
The canonicalizer only deletes unused ops it knows are **side-effect free**.
For all MLIR knows, `alex.transpose` might print something or write memory.
You tell it otherwise with the `Pure` trait:

**`include/Alex/AlexOps.td`**:

```tablegen
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"     // new
include "Alex/AlexDialect.td"
...
def transposeOp: Alex_Op<"transpose", [Pure]>{        // [Pure] added
```

**`include/Alex/AlexOps.h`**:

```cpp
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"     // new
```

`Pure` means "no side effects, and always safe to execute". It also enables
CSE (two identical transposes become one) and hoisting out of loops. Almost
all of your tensor ops (`relu`, `add`, `argmin`, ...) are pure too. Adding
`[Pure]` to them is a good follow-up exercise.

### Test

**`test/transpose/transpose_canonicalize.mlir`** (new):

```mlir
// RUN: alex-opt %s -canonicalize | FileCheck %s

// Folder: swapping a dim with itself is a no-op.
// CHECK-LABEL: func.func @same_dim
// CHECK-SAME: (%[[X:.*]]: tensor<2x3xf32>)
// CHECK-NOT: alex.transpose
// CHECK: return %[[X]]
func.func @same_dim(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %0 = "alex.transpose"(%arg0) {dim0 = 1 : i32, dim1 = -1 : i32}
      : (tensor<2x3xf32>) -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}

// Folder: transposing twice with the same swap gives the input back,
// even when the dims are written differently (1, 0) vs (0, -1).
// CHECK-LABEL: func.func @double_transpose
// CHECK-SAME: (%[[X:.*]]: tensor<2x3xf32>)
// CHECK-NOT: alex.transpose
// CHECK: return %[[X]]
func.func @double_transpose(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %0 = "alex.transpose"(%arg0) {dim0 = 1 : i32, dim1 = 0 : i32}
      : (tensor<2x3xf32>) -> tensor<3x2xf32>
  %1 = "alex.transpose"(%0) {dim0 = 0 : i32, dim1 = -1 : i32}
      : (tensor<3x2xf32>) -> tensor<2x3xf32>
  return %1 : tensor<2x3xf32>
}

// Different swaps must NOT fold.
// CHECK-LABEL: func.func @different_swaps
// CHECK: alex.transpose
// CHECK: alex.transpose
func.func @different_swaps(%arg0: tensor<2x3x4xf32>) -> tensor<3x4x2xf32> {
  %0 = "alex.transpose"(%arg0) {dim0 = 0 : i32, dim1 = 1 : i32}
      : (tensor<2x3x4xf32>) -> tensor<3x2x4xf32>
  %1 = "alex.transpose"(%0) {dim0 = 1 : i32, dim1 = 2 : i32}
      : (tensor<3x2x4xf32>) -> tensor<3x4x2xf32>
  return %1 : tensor<3x4x2xf32>
}

// Canonicalization pattern (uses the custom builder):
// (-1, 0) on rank 3 is rewritten to (0, 2).
// CHECK-LABEL: func.func @normalize_dims
// CHECK: "alex.transpose"(%{{.*}}) <{dim0 = 0 : i32, dim1 = 2 : i32}>
// CHECK-SAME: (tensor<2x3x4xf32>) -> tensor<4x3x2xf32>
func.func @normalize_dims(%arg0: tensor<2x3x4xf32>) -> tensor<4x3x2xf32> {
  %0 = "alex.transpose"(%arg0) {dim0 = -1 : i32, dim1 = 0 : i32}
      : (tensor<2x3x4xf32>) -> tensor<4x3x2xf32>
  return %0 : tensor<4x3x2xf32>
}
```

Things to notice:

- **`@different_swaps` is a negative case.** A folder that is too eager (one
  that folds *any* double transpose) would still pass the first two tests.
  This one catches it: swapping (0,1) then (1,2) is a rotation, not the identity.
- **`@double_transpose` writes the dims differently on purpose**: (1, 0) and
  (0, -1). That checks the folder compares *normalized* dims, not raw attributes.
- **`<{dim0 = ...}>`**: attributes print in `<{ }>` because they're
  *properties* (stored inline in the op). You can still *write* them with
  `{ }` in the input.
- If you remove `[Pure]`, `@double_transpose` fails on its `CHECK-NOT` line.
  That's the dead-op problem from above, and a good way to see why the test
  is written that way.

---

## Step 5: Simplify the lowering

### Why

Before the verifier existed, the lowering had to defend itself: it re-checked
the dims and the result shape, because nothing else did. Now the verifier
guarantees every `alex.transpose` that reaches the lowering is valid. The
duplicate checks are dead code, and dead code drifts out of date.

### Code

In `ConvertTransposeOp::matchAndRewrite`
(`lib/Conversion/AlexToArith/AlexToLinalg.cpp`), replace everything from
`int64_t rank = inputType.getRank();` down to (but not including)
`mlir::Type elementType = ...` with:

```cpp
    int64_t rank = inputType.getRank();

    // The verifier already checked dims and result shape, so normalizing
    // cannot fail here.
    int64_t dim0 = *alex::normalizeDim(op.getDim0Attr().getInt(), rank);
    int64_t dim1 = *alex::normalizeDim(op.getDim1Attr().getInt(), rank);

    // perm[i] = which input dimension lands in output dimension i.
    // Identity with dim0 and dim1 exchanged.
    llvm::SmallVector<unsigned> perm;
    for (int64_t i = 0; i < rank; ++i)
      perm.push_back(i);
    std::swap(perm[dim0], perm[dim1]);

```

What goes away: the manual negative-dim handling, the "out of range" failure,
and the result rank/element-type/shape loop. What stays: the `dyn_cast` +
"must be ranked" check at the top. The verifier *accepts* unranked tensors,
but `linalg.generic` can't handle them, so that check is still about the
lowering's own limits.

### Things to notice

- **A verifier check vs. a lowering check.** The verifier answers "is this op
  valid?". `notifyMatchFailure` in a pattern answers "is this valid op one
  *this pattern* can handle?" Unranked is valid but unsupported, so it stays
  in the pattern. Wrong result shape is invalid, so it moves to the verifier.
- `alex::normalizeDim` is available here because `AlexToLinalg.cpp` already
  includes `Alex/AlexOps.h`. That's the step 1 dependency direction working
  as intended.

Run all three transpose tests again. They should still pass, and
`test/transpose/transpose.mlir` shouldn't need any change.

---

## Checklist

- [ ] Step 1: `normalizeDim` + `inferTransposeType` declared in `AlexOps.h`, defined in `AlexOps.cpp`
- [ ] Step 2: `hasVerifier = 1`, `verify()`, `transpose_invalid.mlir` passes
- [ ] Step 3: `builders = [...]`, custom `build(...)` forwards to the generated one
- [ ] Step 4: `hasFolder`, `hasCanonicalizer`, `[Pure]` + both includes, `PatternMatch.h` include, `transpose_canonicalize.mlir` passes
- [ ] Step 5: lowering trimmed, `transpose.mlir` still passes
- [ ] `lit -v test/transpose` shows 3 PASS

## Follow-up exercises

1. Add `[Pure]` to `relu`, `add`, `argmin` and the other tensor ops. Why is it
   safe for all of them?
2. Add a verifier to `alex.relu`: the result type must equal the input type.
   Write its `transpose_invalid.mlir`-style negative test.
3. Add a folder to `alex.relu`: `relu(relu(x))` → `relu(x)`. (Careful: what do
   you return here? It isn't `x`.)
4. Look up `InferTypeOpInterface`. It would let you delete the custom builder
   from step 3 and also let you write the op without spelling out the result
   type. What would `inferReturnTypes` look like for transpose?
