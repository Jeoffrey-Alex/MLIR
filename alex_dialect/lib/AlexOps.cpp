#include "Alex/AlexOps.h"

#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"

#define GET_OP_CLASSES
#include "Alex/AlexOps.cpp.inc"

mlir::FailureOr<int64_t> alex::normalizeDim(int64_t dim, int64_t rank) {
  // 1. Negative dims count from the end: -1 -> rank - 1.
  if (dim < 0)
    dim += rank;

  // 2. Still outside 0..rank-1? Then the dim doesn't exist.
  if (dim < 0 || dim >= rank)
    return mlir::failure();

  // 3. Valid: the int64_t converts to FailureOr<int64_t> automatically.
  return dim;
}

mlir::FailureOr<mlir::RankedTensorType>
alex::inferTransposeType(mlir::RankedTensorType inputType, int64_t dim0,
                         int64_t dim1) {
  // 1. How many dims the input has.
  int64_t rank = inputType.getRank();

  // 2. Normalize both dims; give up if either is out of range.
  mlir::FailureOr<int64_t> d0 = normalizeDim(dim0, rank);
  mlir::FailureOr<int64_t> d1 = normalizeDim(dim1, rank);
  if (mlir::failed(d0) || mlir::failed(d1))
    return mlir::failure();

  // 3. Copy the shape. getShape() is a read-only view (ArrayRef), so we need
  //    our own SmallVector to modify.
  llvm::SmallVector<int64_t> shape(inputType.getShape());

  // 4. Swap the two dims. *d0 unwraps the number inside the FailureOr.
  std::swap(shape[*d0], shape[*d1]);

  // 5. Same element type, new shape.
  return mlir::RankedTensorType::get(shape, inputType.getElementType());
}

// Declared by `let hasVerifier = 1;` in AlexOps.td. MLIR calls this after
// parsing and after every pass.
mlir::LogicalResult alex::transposeOp::verify() {
  // 1. We are inside the op class, so getInput()/getResult() work directly.
  mlir::RankedTensorType inputType =
      llvm::dyn_cast<mlir::RankedTensorType>(getInput().getType());
  mlir::RankedTensorType resultType =
      llvm::dyn_cast<mlir::RankedTensorType>(getResult().getType());

  // 2. Unranked tensor (tensor<*xf32>): no shape to compare, so accept.
  if (!inputType || !resultType)
    return mlir::success();

  // 3. Read the dims as signed values. getDim0() returns uint32_t and would
  //    turn -1 into 4294967295.
  int64_t dim0 = getDim0Attr().getInt();
  int64_t dim1 = getDim1Attr().getInt();

  // 4. Ask the helper what the result type should be.
  mlir::FailureOr<mlir::RankedTensorType> expected =
      inferTransposeType(inputType, dim0, dim1);
  if (mlir::failed(expected))
    return emitOpError() << "dims (" << dim0 << ", " << dim1
                         << ") out of range for rank " << inputType.getRank();

  // 5. Compare with the result type actually written in the IR.
  if (resultType != *expected)
    return emitOpError() << "expected result type " << *expected << ", got "
                         << resultType;

  return mlir::success();
}

// Declared by `let builders = [...]` in AlexOps.td.
// Lets C++ create a transpose without working out the result type.
void alex::transposeOp::build(mlir::OpBuilder &builder,
                              mlir::OperationState &state, mlir::Value input,
                              int32_t dim0, int32_t dim1) {
  // 1. The input's type, e.g. tensor<2x3xf32>.
  mlir::RankedTensorType inputType =
      llvm::cast<mlir::RankedTensorType>(input.getType());

  // 2. Work out the result type: 2x3 -> 3x2. Same function as the verifier.
  mlir::FailureOr<mlir::RankedTensorType> resultType =
      inferTransposeType(inputType, dim0, dim1);
  assert(mlir::succeeded(resultType) && "transpose dims out of range");

  // 3. Hand everything to the builder ODS generated, which wants the result
  //    type first and the dims as attributes.
  build(builder, state, *resultType, input, builder.getI32IntegerAttr(dim0),
        builder.getI32IntegerAttr(dim1));
}

// Declared by `let hasFolder = 1;` in AlexOps.td.
// transpose(transpose(x)) with the same swap is just x.
mlir::OpFoldResult alex::transposeOp::fold(FoldAdaptor adaptor) {
  mlir::RankedTensorType inputType =
      llvm::dyn_cast<mlir::RankedTensorType>(getInput().getType());
  if (!inputType)
    return {};
  int64_t rank = inputType.getRank();

  // 1. My two dims, cleaned up. The verifier already checked them, so * is
  // safe.
  int64_t dim0 = *normalizeDim(getDim0Attr().getInt(), rank);
  int64_t dim1 = *normalizeDim(getDim1Attr().getInt(), rank);

  // 2. Was my input made by another transpose?
  transposeOp inner = getInput().getDefiningOp<transposeOp>();
  if (!inner)
    return {};

  // 3. Did it swap the same two dims as me?
  int64_t innerDim0 = *normalizeDim(inner.getDim0Attr().getInt(), rank);
  int64_t innerDim1 = *normalizeDim(inner.getDim1Attr().getInt(), rank);
  bool sameSwap = (innerDim0 == dim0 && innerDim1 == dim1) ||
                  (innerDim0 == dim1 && innerDim1 == dim0);
  if (!sameSwap)
    return {};

  // 4. Swap undone: replace me with the inner transpose's input (the original
  // x).
  return inner.getInput();
}

namespace {

/// Rewrites dims into standard form: not negative, and dim0 < dim1.
/// 2x3 with (-1, 0) -> (0, 1)
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

    // 1. The dims as the user wrote them, e.g. -1 and 0.
    int64_t dim0 = op.getDim0Attr().getInt();
    int64_t dim1 = op.getDim1Attr().getInt();

    // 2. Clean them up (-1 -> 1) and put the smaller one first: lo = 0, hi = 1.
    int64_t a = *alex::normalizeDim(dim0, rank);
    int64_t b = *alex::normalizeDim(dim1, rank);
    int64_t lo = std::min(a, b);
    int64_t hi = std::max(a, b);

    // 3. Already in standard form? Then there is nothing to do.
    if (dim0 == lo && dim1 == hi)
      return mlir::failure();

    // 4. Replace the op with a new transpose using the clean dims.
    //    This calls YOUR builder: (input, int32_t, int32_t).
    rewriter.replaceOpWithNewOp<alex::transposeOp>(op, op.getInput(), lo, hi);
    return mlir::success();
  }
};

} // namespace

// Declared by `let hasCanonicalizer = 1;` in AlexOps.td.
void alex::transposeOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet &results, mlir::MLIRContext *context) {
  results.add<NormalizeTransposeDims>(context);
}

// for relu operations
mlir::LogicalResult alex::reluOp::verify() {
  mlir::Type inputType = getInput().getType();
  mlir::Type resultType = getResult().getType();

  if (inputType != resultType)
    return emitOpError() << "expected result type " << inputType << ", got"
                         << resultType;

  return mlir::success();
}

mlir::OpFoldResult alex::reluOp::fold(FoldAdaptor adaptor) {
  reluOp inner = getInput().getDefiningOp<reluOp>();

  if (!inner)
    return {};

  return getInput();
}

namespace {

/// Moves relu above transpose: relu(transpose(x)) -> transpose(relu(x)).
struct SwapReluTranspose : public mlir::OpRewritePattern<alex::reluOp> {
  using mlir::OpRewritePattern<alex::reluOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(alex::reluOp op,
                  mlir::PatternRewriter &rewriter) const override {
    alex::transposeOp transpose =
        op.getInput().getDefiningOp<alex::transposeOp>();

    if (!transpose)
      return mlir::failure();

    // If the transpose has other users it stays alive, so the swap would only
    // add ops.
    if (!transpose.getResult().hasOneUse())
      return mlir::failure();

    mlir::Value x = transpose.getInput();

    alex::reluOp newRelu =
        alex::reluOp::create(rewriter, op.getLoc(), x.getType(), x);

    int32_t dim0 = static_cast<int32_t>(transpose.getDim0Attr().getInt());
    int32_t dim1 = static_cast<int32_t>(transpose.getDim1Attr().getInt());
    rewriter.replaceOpWithNewOp<alex::transposeOp>(op, newRelu.getResult(),
                                                   dim0, dim1);

    return mlir::success();
  }
};

} // namespace

void alex::reluOp::getCanonicalizationPatterns(mlir::RewritePatternSet &results,
                                               mlir::MLIRContext *context) {
  results.add<SwapReluTranspose>(context);
}