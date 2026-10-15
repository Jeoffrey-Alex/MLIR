# Lowering `alex.argmin` → Linalg

This note explains how `ConvertArgminOp` in [AlexToLinalg.cpp](AlexToLinalg.cpp)
turns `alex.argmin` into a `linalg.generic`.

Run it with:

```bash
alex-opt --convert-alex-to-linalg test/python/argmin/argmin_dim_none.mlir
```

---

## 1. What argmin does

`argmin` returns the **position of the smallest value** in a tensor.

```
input = [[1, 2, 3],
         [4, 0, 6],     smallest value is 0
         [7, 8, 9]]     at row 1, col 1

flattened: [1, 2, 3, 4, 0, 6, 7, 8, 9]
                        ^
                     index 4   →  result = 4
```

With no `dim`, the whole tensor is treated as one flat list, and the result is a
single `i64` (a 0-D tensor, `tensor<i64>`).

---

## 2. The idea in plain words

Imagine walking over every element with two "notes" in your pocket:

| Note           | Starts as                          | Meaning                  |
|----------------|------------------------------------|--------------------------|
| `currentMin`   | the largest possible value (`+inf` for floats, `INT_MAX` for ints) | smallest value seen so far |
| `currentIndex` | `0`                                | where we saw it          |

For each element:

1. Work out its flat index.
2. If `value < currentMin`, update both notes.
3. Otherwise, keep the notes as they are.

At the end, `currentIndex` is the answer.

That is exactly what the lowering builds, using Linalg pieces.

---

## 3. Step by step through the code

### Step 1 — Check the input is supported

```cpp
if (!inputType || !inputType.hasStaticShape())  -> fail
if (op.getDim())                                -> fail  ("dim not supported yet")
if (rank == 0)                                  -> fail
```

So the pattern only handles: **ranked, static-shape tensor, rank ≥ 1, no `dim`.**

### Step 2 — Make the two starting "notes"

```mlir
%max  = arith.constant 2147483647 : i32     // INT_MAX (or +inf for floats)
%zero = arith.constant 0 : i64

%0 = tensor.empty() : tensor<i32>           // holder for currentMin
%1 = tensor.empty() : tensor<i64>           // holder for currentIndex
%2 = linalg.fill ins(%max)  outs(%0)        // currentMin   = INT_MAX
%3 = linalg.fill ins(%zero) outs(%1)        // currentIndex = 0
```

Both are 0-D tensors (a single value each), because the result is one number.

### Step 3 — Describe the loop shape (indexing maps)

For a 2-D input the loop variables are `(d0, d1)` = (row, col).

| Operand      | Map                         | Meaning                         |
|--------------|-----------------------------|---------------------------------|
| input        | `(d0, d1) -> (d0, d1)`      | read element `[row, col]`       |
| currentMin   | `(d0, d1) -> ()`            | always the same single slot     |
| currentIndex | `(d0, d1) -> ()`            | always the same single slot     |

Every dimension is marked `"reduction"`, because all of them collapse into one
output value.

### Step 4 — The loop body (what happens per element)

```mlir
^bb0(%in: i32, %out: i32, %out_0: i64):     // value, currentMin, currentIndex
  // flat index = row * 3 + col * 1
  %col  = linalg.index 1
  %t1   = arith.muli %col, %c1
  %row  = linalg.index 0
  %t2   = arith.muli %row, %c3
  %flat = arith.addi %t1, %t2
  %idx  = arith.index_cast %flat : index to i64

  %smaller  = arith.cmpi slt, %in, %out     // value < currentMin ?
  %newMin   = arith.select %smaller, %in,  %out
  %newIndex = arith.select %smaller, %idx, %out_0
  linalg.yield %newMin, %newIndex
```

**Flat index formula (row-major):** walk dimensions from last to first,
multiplying each loop index by the product of the sizes to its right.

```
shape 3x3     → flat = d0*3 + d1*1
shape 2x3x4   → flat = d0*12 + d1*4 + d2*1
```

**Comparison:** `arith.cmpf olt` for floats, `arith.cmpi slt` for integers.
Strictly-less-than means that on a tie the **first** occurrence wins (same as
PyTorch).

### Step 5 — Return only the index

`linalg.generic` produces two results: `(min value, index)`.
We only need the index, so the op is replaced with result `#1`:

```cpp
rewriter.replaceOp(op, genericOp.getResult(1));
```

```mlir
return %4#1 : tensor<i64>
```

The min-value result is unused and will be removed by later cleanup passes.

---

## 4. Full before / after

**Before**

```mlir
%result = "alex.argmin"(%input) : (tensor<3x3xi32>) -> tensor<i64>
```

**After**

```mlir
#map  = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> ()>

%c2147483647_i32 = arith.constant 2147483647 : i32
%c0_i64 = arith.constant 0 : i64
%0 = tensor.empty() : tensor<i32>
%1 = tensor.empty() : tensor<i64>
%2 = linalg.fill ins(%c2147483647_i32 : i32) outs(%0 : tensor<i32>) -> tensor<i32>
%3 = linalg.fill ins(%c0_i64 : i64) outs(%1 : tensor<i64>) -> tensor<i64>
%4:2 = linalg.generic {indexing_maps = [#map, #map1, #map1],
                       iterator_types = ["reduction", "reduction"]}
    ins(%cst : tensor<3x3xi32>) outs(%2, %3 : tensor<i32>, tensor<i64>) {
^bb0(%in: i32, %out: i32, %out_0: i64):
  %5 = linalg.index 1 : index
  %c1 = arith.constant 1 : index
  %6 = arith.muli %5, %c1 : index
  %7 = linalg.index 0 : index
  %c3 = arith.constant 3 : index
  %8 = arith.muli %7, %c3 : index
  %9 = arith.addi %6, %8 : index
  %10 = arith.index_cast %9 : index to i64
  %11 = arith.cmpi slt, %in, %out : i32
  %12 = arith.select %11, %in, %out : i32
  %13 = arith.select %11, %10, %out_0 : i64
  linalg.yield %12, %13 : i32, i64
} -> (tensor<i32>, tensor<i64>)
return %4#1 : tensor<i64>
```

(This is the real output of the pass on `test/python/argmin/argmin_dim_none.mlir`.)

---

## 5. Current limitations

| Case                         | Supported? |
|------------------------------|------------|
| No `dim` (whole tensor)      | ✅ |
| `dim = N`                    | ❌ fails with "failed to legalize operation 'alex.argmin'" |
| `keepdim = true`             | ❌ (only matters with `dim`) |
| Dynamic shapes (`tensor<?x3xi32>`) | ❌ |
| Rank-0 input                 | ❌ |

**Ideas for adding `dim` support:** mark only dimension `dim` as `"reduction"`
(others `"parallel"`), drop `dim` from the output indexing map, use
`linalg.index dim` directly as the index (no flattening), and make the init
tensors have the reduced shape (with a size-1 dim when `keepdim = true`).

NaN note: `olt` is false whenever NaN is involved, so NaNs are skipped. PyTorch
instead returns the index of a NaN if one exists.
