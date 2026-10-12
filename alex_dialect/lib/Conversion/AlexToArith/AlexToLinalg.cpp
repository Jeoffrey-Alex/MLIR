#include "Alex/AlexOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include <functional>
#include <iostream>

namespace {

class ConvertReductionOp
    : public mlir::OpConversionPattern<alex::reductionLinalgOp> {
public:
  using mlir::OpConversionPattern<alex::reductionLinalgOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(alex::reductionLinalgOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();

    mlir::RankedTensorType inputType =
        llvm::dyn_cast<mlir::RankedTensorType>(adaptor.getInput1().getType());

    if (!inputType)
      return rewriter.notifyMatchFailure(op, "input must be a ranked tensor");

    mlir::Type elementType = inputType.getElementType();
    if (elementType != op.getResult().getType())
      return rewriter.notifyMatchFailure(
          op, "result type must match the input element type");

    // Temporary 0-D output tensor holding the accumulator.
    mlir::RankedTensorType outputType =
        mlir::RankedTensorType::get({}, elementType);

    mlir::tensor::EmptyOp empty = mlir::tensor::EmptyOp::create(
        rewriter, loc, outputType.getShape(), elementType);

    // tensor.empty is uninitialized, so seed the accumulator with 0.
    mlir::Value zero = mlir::arith::ConstantOp::create(
        rewriter, loc, elementType,
        llvm::cast<mlir::TypedAttr>(rewriter.getZeroAttr(elementType)));
    mlir::Value init =
        mlir::linalg::FillOp::create(rewriter, loc, mlir::ValueRange{zero},
                                     mlir::ValueRange{empty.getResult()})
            .getResult(0);

    // Reduce over every dimension of the input.
    unsigned rank = inputType.getRank();

    llvm::SmallVector<mlir::utils::IteratorType> iteratorTypes(
        rank, mlir::utils::IteratorType::reduction);

    mlir::AffineMap inputMap =
        mlir::AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());

    mlir::AffineMap outputMap =
        mlir::AffineMap::get(rank, 0, {}, rewriter.getContext());

    llvm::SmallVector<mlir::AffineMap> indexingMaps = {inputMap, outputMap};

    mlir::linalg::GenericOp genericOp = mlir::linalg::GenericOp::create(
        rewriter, loc, mlir::TypeRange{outputType},
        mlir::ValueRange{adaptor.getInput1()}, mlir::ValueRange{init},
        indexingMaps, iteratorTypes);

    mlir::Region &region = genericOp.getRegion();
    mlir::Block *block = new mlir::Block();
    region.push_back(block);

    // Input element and accumulator element
    block->addArguments(elementType, loc);
    block->addArguments(elementType, loc);

    mlir::Value arg0 = block->getArgument(0);
    mlir::Value arg1 = block->getArgument(1);

    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(block);

    mlir::Value result;
    if (llvm::isa<mlir::FloatType>(elementType))
      result = mlir::arith::AddFOp::create(rewriter, loc, arg0, arg1);
    else
      result = mlir::arith::AddIOp::create(rewriter, loc, arg0, arg1);

    mlir::linalg::YieldOp::create(rewriter, loc, result);

    rewriter.setInsertionPointAfter(genericOp);

    mlir::tensor::ExtractOp scalarResult = mlir::tensor::ExtractOp::create(
        rewriter, loc, genericOp.getResult(0), mlir::ValueRange{});

    rewriter.replaceOp(op, scalarResult.getResult());

    return mlir::success();
  }
};

/*
//addition of two linalg ops to linalg.generic op
class ConvertAccAddDivOp : public mlir::OpConversionPattern<alex::addLinalgOp> {
public:
using mlir::OpConversionPattern<alex::addLinalgOp>::OpConversionPattern;

mlir::LogicalResult
matchAndRewrite(alex::addLinalgOp op, OpAdaptor adaptor,
              mlir::ConversionPatternRewriter &rewriter) const override {
mlir::Location loc = op.getLoc();

// result type =  tensor<2x2xf32>
mlir::RankedTensorType resultType =
    llvm::cast<mlir::RankedTensorType>(op.getResult().getType());

mlir::Type elementType = resultType.getElementType();

//use mlir::Value empty [you dont have to use .getResult() because the empty op
returns a single value] mlir::tensor::EmptyOp empty =
mlir::tensor::EmptyOp::create( rewriter, loc, resultType.getShape(),
resultType.getElementType());

// Iterator Types
llvm::SmallVector<mlir::utils::IteratorType> IteratorTypes = {
    mlir::utils::IteratorType::parallel,
    mlir::utils::IteratorType::parallel};

mlir::AffineMap identityMap = mlir::AffineMap::getMultiDimIdentityMap(
    resultType.getRank(), rewriter.getContext());

mlir::SmallVector<mlir::AffineMap> indexingMaps = {identityMap, identityMap,
                                                   identityMap};

mlir::linalg::GenericOp genericOp = mlir::linalg::GenericOp::create(
    rewriter, loc, mlir::TypeRange{resultType},
    mlir::ValueRange{adaptor.getInput1(), adaptor.getInput2()},
    mlir::ValueRange{empty.getResult()}, indexingMaps, IteratorTypes);

mlir::Region &region = genericOp.getRegion();
mlir::Block *block = new mlir::Block();
region.push_back(block);

block->addArguments(elementType, loc);
block->addArguments(elementType, loc);
block->addArguments(elementType, loc);

mlir::Value arg0 = block->getArgument(0);
mlir::Value arg1 = block->getArgument(1);

mlir::OpBuilder::InsertionGuard guard(rewriter);
rewriter.setInsertionPointToStart(block);

mlir::Value result = mlir::arith::AddFOp::create(rewriter, loc, arg0, arg1);

mlir::linalg::YieldOp::create(rewriter, loc, result);

rewriter.replaceOp(op, genericOp.getResults());

return mlir::success();
}
};
*/

class ConvertAccAddDivOp : public mlir::OpConversionPattern<alex::addLinalgOp> {
public:
  using mlir::OpConversionPattern<alex::addLinalgOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(alex::addLinalgOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();

    mlir::RankedTensorType resultType =
        llvm::cast<mlir::RankedTensorType>(op.getType());

    // 1. Create empty output tensor.
    mlir::tensor::EmptyOp empty = mlir::tensor::EmptyOp::create(
        rewriter, loc, resultType.getShape(), resultType.getElementType());

    // 2. Create kind attribute.
    mlir::linalg::ElementwiseKindAttr kindAttr =
        mlir::linalg::ElementwiseKindAttr::get(
            rewriter.getContext(), mlir::linalg::ElementwiseKind::add);

    // 4. Create identity indexing maps.
    mlir::AffineMap identityMap = mlir::AffineMap::getMultiDimIdentityMap(
        resultType.getRank(), rewriter.getContext());

    mlir::ArrayAttr indexingMaps =
        rewriter.getAffineMapArrayAttr({identityMap, identityMap, identityMap});

    // 5. Create linalg.elementwise.
    mlir::linalg::ElementwiseOp newOp = mlir::linalg::ElementwiseOp::create(
        rewriter, loc,
        mlir::ValueRange{adaptor.getInput1(), adaptor.getInput2()},
        mlir::ValueRange{empty.getResult()}, kindAttr, indexingMaps);

    // 6. Replace original op.
    rewriter.replaceOp(op, newOp.getResults());

    return mlir::success();
  }
};

class ConvertaddBCLinalgOp
    : public mlir::OpConversionPattern<alex::addBCLinalgOp> {
public:
  using mlir::OpConversionPattern<alex::addBCLinalgOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(alex::addBCLinalgOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {

    mlir::Location loc = op.getLoc();

    mlir::RankedTensorType input1Type =
        llvm::cast<mlir::RankedTensorType>(adaptor.getInput1().getType());

    mlir::RankedTensorType input2Type =
        llvm::cast<mlir::RankedTensorType>(adaptor.getInput2().getType());

    mlir::RankedTensorType resultType =
        llvm::cast<mlir::RankedTensorType>(op.getType());

    // 1. Calculate broadcasted shape.
    llvm::SmallVector<int64_t> broadcastShape;

    if (!mlir::OpTrait::util::getBroadcastedShape(
            input1Type.getShape(), input2Type.getShape(), broadcastShape)) {
      return rewriter.notifyMatchFailure(op,
                                         "input tensors are not broadcastable");
    }

    // 2. Create output/destination tensor.
    mlir::tensor::EmptyOp empty = mlir::tensor::EmptyOp::create(
        rewriter, loc, broadcastShape, resultType.getElementType());

    // 3. Create kind = add.
    mlir::linalg::ElementwiseKindAttr kindAttr =
        mlir::linalg::ElementwiseKindAttr::get(
            rewriter.getContext(), mlir::linalg::ElementwiseKind::add);

    // 4. Create indexing map for each operand.
    auto getBroadcastMap =
        [&](mlir::ArrayRef<int64_t> operandShape) -> mlir::AffineMap {
      unsigned finalRank = broadcastShape.size();
      unsigned operandRank = operandShape.size();

      llvm::SmallVector<mlir::AffineExpr> expressions;

      for (unsigned i = 0; i < operandRank; ++i) {
        unsigned finalDim = finalRank - operandRank + i;

        if (operandShape[i] == 1) {
          expressions.push_back(
              mlir::getAffineConstantExpr(0, rewriter.getContext()));
        } else {
          expressions.push_back(
              mlir::getAffineDimExpr(finalDim, rewriter.getContext()));
        }
      }

      return mlir::AffineMap::get(finalRank, 0, expressions,
                                  rewriter.getContext());
    };

    mlir::AffineMap input1Map = getBroadcastMap(input1Type.getShape());

    mlir::AffineMap input2Map = getBroadcastMap(input2Type.getShape());

    mlir::AffineMap outputMap = mlir::AffineMap::getMultiDimIdentityMap(
        broadcastShape.size(), rewriter.getContext());

    // 5. Create indexing_maps attribute.
    mlir::ArrayAttr indexingMaps =
        rewriter.getAffineMapArrayAttr({input1Map, input2Map, outputMap});

    // 6. Create linalg.elementwise.
    mlir::linalg::ElementwiseOp newOp = mlir::linalg::ElementwiseOp::create(
        rewriter, loc,
        mlir::ValueRange{adaptor.getInput1(), adaptor.getInput2()},
        mlir::ValueRange{empty.getResult()}, kindAttr, indexingMaps);

    // 7. Replace Alex op.
    rewriter.replaceOp(op, newOp.getResults());

    return mlir::success();
  }
};

} // namespace

class AlexToLinalgPass
    : public mlir::PassWrapper<AlexToLinalgPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AlexToLinalgPass)

  mlir::StringRef getArgument() const final { return "convert-alex-to-linalg"; }

  mlir::StringRef getDescription() const final {
    return "Lower Alex operations to Linalg operations";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::linalg::LinalgDialect, mlir::arith::ArithDialect,
                    mlir::tensor::TensorDialect>();
  }

  void runOnOperation() override {
    mlir::MLIRContext *context = &getContext();
    mlir::ModuleOp module = getOperation();

    mlir::ConversionTarget target(*context);

    target.addLegalDialect<mlir::linalg::LinalgDialect,
                           mlir::tensor::TensorDialect,
                           mlir::arith::ArithDialect>();

    target.addIllegalOp<alex::addLinalgOp, alex::addBCLinalgOp,
                        alex::reductionLinalgOp>();

    mlir::RewritePatternSet patterns(context);
    patterns.add<ConvertAccAddDivOp, ConvertReductionOp, ConvertaddBCLinalgOp>(
        context);

    if (failed(mlir::applyPartialConversion(module, target,
                                            std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<mlir::Pass> createAlexToLinalgPass() {
  return std::make_unique<AlexToLinalgPass>();
}
