#include "Alex/AlexOps.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace {
// Decompose alex.addcmul into simpler alex ops: result = input + value
// *(tensor1 * tensor2)

class DecomposeAddcmulPattern : public mlir::OpRewritePattern<alex::AddcmulOp> {
public:
  using mlir::OpRewritePattern<alex::AddcmulOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(alex::AddcmulOp op,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();

    // We need ranked tensors to compute shapes and build a splat constant
    mlir::RankedTensorType inputType =
        llvm::dyn_cast<mlir::RankedTensorType>(op.getInput().getType());
    mlir::RankedTensorType tensor1Type =
        llvm::dyn_cast<mlir::RankedTensorType>(op.getTensor1().getType());
    mlir::RankedTensorType tensor2Type =
        llvm::dyn_cast<mlir::RankedTensorType>(op.getTensor2().getType());
    mlir::RankedTensorType resultType =
        llvm::dyn_cast<mlir::RankedTensorType>(op.getResult().getType());
    if (!inputType || !tensor1Type || !tensor2Type || !resultType)
      return rewriter.notifyMatchFailure(op, "all types must be ranked tensors");

    // All element types must match, only shapes may differ
    mlir::Type elementType = resultType.getElementType();
    if (inputType.getElementType() != elementType ||
        tensor1Type.getElementType() != elementType ||
        tensor2Type.getElementType() != elementType)
      return rewriter.notifyMatchFailure(op, "element types must match");

    // Shape of tensor1 * tensor2 after broadcasting
    llvm::SmallVector<int64_t> productShape;
    if (!mlir::OpTrait::util::getBroadcastedShape(
            tensor1Type.getShape(), tensor2Type.getShape(), productShape))
      return rewriter.notifyMatchFailure(
          op, "tensor1 and tensor2 shapes are not broadcastable");

    // Shape of input + product after broadcasting, must be the result shape
    llvm::SmallVector<int64_t> finalShape;
    if (!mlir::OpTrait::util::getBroadcastedShape(inputType.getShape(),
                                                  productShape, finalShape) ||
        llvm::ArrayRef<int64_t>(finalShape) != resultType.getShape())
      return rewriter.notifyMatchFailure(
          op, "broadcasted shape does not match result shape");

    // Type of tensor1 * tensor2, and of value * (tensor1 * tensor2)
    mlir::RankedTensorType productType =
        mlir::RankedTensorType::get(productShape, elementType);

    // Build the scalar attribute for 'value' in the tensor's element type
    double scale = op.getValue().convertToDouble();

    mlir::TypedAttr scalarAttr;
    if (llvm::isa<mlir::FloatType>(elementType))
      scalarAttr = rewriter.getFloatAttr(elementType, scale);
    else if (llvm::isa<mlir::IntegerType>(elementType))
      scalarAttr =
          rewriter.getIntegerAttr(elementType, static_cast<int64_t>(scale));
    else
      return rewriter.notifyMatchFailure(op, "unsupported element type");

    // splat: a tensor where every element is 'scale'.
    mlir::DenseElementsAttr splatAttr =
        mlir::DenseElementsAttr::get(productType, scalarAttr);
    // prod = tensor1 * tensor2
    alex::MulOp prod = alex::MulOp::create(rewriter, loc, productType,
                                           op.getTensor1(), op.getTensor2());

    // scaleConst = dense<value>
    alex::ConstOp scaleConst =
        alex::ConstOp::create(rewriter, loc, productType, splatAttr);

    // scaled = scaleConst * prod
    alex::MulOp scaled = alex::MulOp::create(
        rewriter, loc, productType, scaleConst.getResult(), prod.getResult());

    // result = input + scaled, and this replaces the addcmul op
    rewriter.replaceOpWithNewOp<alex::AddOp>(op, resultType, op.getInput(),
                                             scaled.getResult());

    return mlir::success();
  }
};

// Pass that runs the addcmul decomposition pattern.
class DecomposeAddcmulPass
    : public mlir::PassWrapper<DecomposeAddcmulPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DecomposeAddcmulPass)

  // Pass name
  mlir::StringRef getArgument() const final {
    return "alex-decompose-addcmul";
  }

  // Pass description
  mlir::StringRef getDescription() const final {
    return "Decompose alex.addcmul into alex.mul, alex.constant and alex.add";
  }

  // Entry point of the pass
  void runOnOperation() override {
    mlir::MLIRContext &context = getContext();

    mlir::RewritePatternSet patterns(&context);
    patterns.add<DecomposeAddcmulPattern>(&context);

    // Apply the patterns repeatedly until nothing more changes.
    if (failed(mlir::applyPatternsGreedily(getOperation(),
                                           std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

// Create and return the addcmul decomposition pass.
std::unique_ptr<mlir::Pass> createDecomposeAddcmulPass() {
  return std::make_unique<DecomposeAddcmulPass>();
}
