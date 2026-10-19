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
//returns a single value]
mlir::tensor::EmptyOp empty =
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

class ConvertArgminOp : public mlir::OpConversionPattern<alex::argminOp> {
public:
  using mlir::OpConversionPattern<alex::argminOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(alex::argminOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    mlir::Value input = adaptor.getInput();
    mlir::RankedTensorType inputType =
        llvm::dyn_cast<mlir::RankedTensorType>(input.getType());

    if (!inputType || !inputType.hasStaticShape())
      return rewriter.notifyMatchFailure(
          op, "input must be a ranked tensor with static shape");
    int64_t rank = inputType.getRank();
    if (rank == 0)
      return rewriter.notifyMatchFailure(
          op, "input tensor must have rank greater than 0");

    // Reduction dimension: -1 means reduce over all dims (dim = None).
    int64_t reduceDim = -1;
    if (std::optional<int32_t> dimAttr = op.getDim()) {
      reduceDim = *dimAttr < 0 ? *dimAttr + rank : *dimAttr;
      if (reduceDim < 0 || reduceDim >= rank)
        return rewriter.notifyMatchFailure(op, "dim is out of range");
    }
    bool reduceAll = reduceDim < 0;

    mlir::Type elementType = inputType.getElementType();
    mlir::IntegerType i64Type = rewriter.getI64Type();

    // Shape after reduction (keepdim = false): drop the reduced dim, or all
    // dims when dim = None. Output map keeps the same (non-reduced) dims.
    llvm::SmallVector<int64_t> resultShape;
    llvm::SmallVector<mlir::AffineExpr> outputExprs;
    if (!reduceAll) {
      for (int64_t d = 0; d < rank; ++d) {
        if (d == reduceDim)
          continue;
        resultShape.push_back(inputType.getDimSize(d));
        outputExprs.push_back(rewriter.getAffineDimExpr(d));
      }
    }
    mlir::RankedTensorType minTensorType =
        mlir::RankedTensorType::get(resultShape, elementType);
    mlir::RankedTensorType indexTensorType =
        mlir::RankedTensorType::get(resultShape, i64Type);

    mlir::TypedAttr maxAttr;
    if (mlir::FloatType floatType =
            llvm::dyn_cast<mlir::FloatType>(elementType))
      maxAttr = rewriter.getFloatAttr(
          floatType, llvm::APFloat::getInf(floatType.getFloatSemantics()));
    else
      maxAttr = rewriter.getIntegerAttr(
          elementType,
          llvm::APInt::getSignedMaxValue(elementType.getIntOrFloatBitWidth()));

    mlir::Value maxValue =
        mlir::arith::ConstantOp::create(rewriter, loc, maxAttr);
    mlir::Value zero = mlir::arith::ConstantOp::create(
        rewriter, loc, rewriter.getI64IntegerAttr(0));

    mlir::tensor::EmptyOp emptyMin =
        mlir::tensor::EmptyOp::create(rewriter, loc, resultShape, elementType);
    mlir::tensor::EmptyOp emptyIndex =
        mlir::tensor::EmptyOp::create(rewriter, loc, resultShape, i64Type);
    mlir::Value initMin =
        mlir::linalg::FillOp::create(rewriter, loc, mlir::ValueRange{maxValue},
                                     mlir::ValueRange{emptyMin.getResult()})
            .getResult(0);
    mlir::Value initIndex =
        mlir::linalg::FillOp::create(rewriter, loc, mlir::ValueRange{zero},
                                     mlir::ValueRange{emptyIndex.getResult()})
            .getResult(0);

    mlir::AffineMap inputMap =
        mlir::AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());
    mlir::AffineMap outputMap =
        mlir::AffineMap::get(rank, 0, outputExprs, rewriter.getContext());
    llvm::SmallVector<mlir::AffineMap> indexingMaps = {inputMap, outputMap,
                                                       outputMap};

    // dim = None: every dim is a reduction. Otherwise only reduceDim is.
    llvm::SmallVector<mlir::utils::IteratorType> iteratorTypes(
        rank, reduceAll ? mlir::utils::IteratorType::reduction
                        : mlir::utils::IteratorType::parallel);
    if (!reduceAll)
      iteratorTypes[reduceDim] = mlir::utils::IteratorType::reduction;

    mlir::linalg::GenericOp genericOp = mlir::linalg::GenericOp::create(
        rewriter, loc, mlir::TypeRange{minTensorType, indexTensorType},
        mlir::ValueRange{input}, mlir::ValueRange{initMin, initIndex},
        indexingMaps, iteratorTypes,
        [&](mlir::OpBuilder &b, mlir::Location l, mlir::ValueRange args) {
          mlir::Value value = args[0];
          mlir::Value currentMin = args[1];
          mlir::Value currentIndex = args[2];

          // dim given: the index along that dim is the answer.
          // dim = None: row-major flattened index over all dims.
          mlir::Value candidateIndex;
          if (!reduceAll)
            candidateIndex = mlir::linalg::IndexOp::create(b, l, reduceDim);
          int64_t stride = 1;
          for (int64_t d = rank - 1; reduceAll && d >= 0; --d) {
            mlir::Value dimIndex = mlir::linalg::IndexOp::create(b, l, d);
            mlir::Value strideValue =
                mlir::arith::ConstantIndexOp::create(b, l, stride);
            mlir::Value term =
                mlir::arith::MulIOp::create(b, l, dimIndex, strideValue);
            candidateIndex =
                candidateIndex
                    ? mlir::arith::AddIOp::create(b, l, candidateIndex, term)
                          .getResult()
                    : term;
            stride *= inputType.getDimSize(d);
          }

          mlir::Value candidateIndexI64 =
              mlir::arith::IndexCastOp::create(b, l, i64Type, candidateIndex);

          mlir::Value isSmaller;
          if (llvm::isa<mlir::FloatType>(elementType))
            isSmaller = mlir::arith::CmpFOp::create(
                b, l, mlir::arith::CmpFPredicate::OLT, value, currentMin);
          else
            isSmaller = mlir::arith::CmpIOp::create(
                b, l, mlir::arith::CmpIPredicate::slt, value, currentMin);

          mlir::Value newMin =
              mlir::arith::SelectOp::create(b, l, isSmaller, value, currentMin);
          mlir::Value newIndex = mlir::arith::SelectOp::create(
              b, l, isSmaller, candidateIndexI64, currentIndex);
          mlir::linalg::YieldOp::create(b, l,
                                        mlir::ValueRange{newMin, newIndex});
        });

    mlir::Value result = genericOp.getResult(1);

    // keepdim = true: put the reduced dim(s) back with size 1.
    if (op.getKeepdim()) {
      llvm::SmallVector<int64_t> keptShape;
      llvm::SmallVector<mlir::ReassociationIndices> reassociation;
      for (int64_t d = 0; d < rank; ++d) {
        bool reduced = reduceAll || d == reduceDim;
        keptShape.push_back(reduced ? 1 : inputType.getDimSize(d));
        if (resultShape.empty())
          continue; // Scalar -> 1x1x..: empty reassociation.
        // Each source dim gets its own group; the size-1 dim joins the group
        // of its left neighbour (or the right one when it is dim 0).
        if (reduced && d > 0)
          reassociation.back().push_back(d);
        else if (reduced)
          reassociation.push_back({d});
        else if (!reassociation.empty() && reduceDim == 0 && d == 1)
          reassociation.back().push_back(d);
        else
          reassociation.push_back({d});
      }
      mlir::RankedTensorType keptType =
          mlir::RankedTensorType::get(keptShape, i64Type);
      result = mlir::tensor::ExpandShapeOp::create(rewriter, loc, keptType,
                                                   result, reassociation);
    }

    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

class ConvertreluOp : public mlir::OpConversionPattern<alex::reluOp> {
public:
  using mlir::OpConversionPattern<alex::reluOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(alex::reluOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    mlir::RankedTensorType inputType =
        llvm::dyn_cast<mlir::RankedTensorType>(adaptor.getInput().getType());
    if (!inputType)
      return rewriter.notifyMatchFailure(op, "input must be a ranked tensor");

    mlir::Type elementType = inputType.getElementType();
    unsigned rank = inputType.getRank();

    mlir::arith::ConstantOp zero = mlir::arith::ConstantOp::create(
        rewriter, loc, rewriter.getZeroAttr(elementType));
    mlir::tensor::EmptyOp empty = mlir::tensor::EmptyOp::create(
        rewriter, loc, inputType.getShape(), elementType);

    llvm::SmallVector<mlir::utils::IteratorType> iteratorTypes(
        rank, mlir::utils::IteratorType::parallel);
    mlir::AffineMap identityMap =
        mlir::AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());
    llvm::SmallVector<mlir::AffineMap> indexingMaps = {identityMap,
                                                       identityMap};

    mlir::linalg::GenericOp genericOp = mlir::linalg::GenericOp::create(
        rewriter, loc, mlir::TypeRange{inputType},
        mlir::ValueRange{adaptor.getInput()},
        mlir::ValueRange{empty.getResult()}, indexingMaps, iteratorTypes,
        [&](mlir::OpBuilder &b, mlir::Location l, mlir::ValueRange args) {
          mlir::Value input = args[0];
          mlir::Value reluResult;

          if (llvm::isa<mlir::FloatType>(elementType))
            reluResult = mlir::arith::MaximumFOp::create(b, l, input, zero); // input > 0.0 ? input : 0.0
          else
            reluResult = mlir::arith::MaxSIOp::create(b, l, input, zero);

          mlir::linalg::YieldOp::create(b, l, reluResult);
        });

    rewriter.replaceOp(op, genericOp.getResult(0));
    return mlir::success();
  }
};

class ConvertTransposeOp
    : public mlir::OpConversionPattern<alex::transposeOp> {
public:
  using mlir::OpConversionPattern<alex::transposeOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(alex::transposeOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();

    mlir::RankedTensorType inputType =
        llvm::dyn_cast<mlir::RankedTensorType>(adaptor.getInput().getType());
    mlir::RankedTensorType resultType =
        llvm::dyn_cast<mlir::RankedTensorType>(op.getResult().getType());
    if (!inputType || !resultType)
      return rewriter.notifyMatchFailure(op, "input and result must be ranked");

    int64_t rank = inputType.getRank();

    // The verifier already checked the dims and the result shape, so
    // normalizing cannot fail here.
    int64_t dim0 = *alex::normalizeDim(op.getDim0Attr().getInt(), rank);
    int64_t dim1 = *alex::normalizeDim(op.getDim1Attr().getInt(), rank);

    // perm[i] = which input dimension lands in output dimension i.
    // Identity with dim0 and dim1 exchanged.
    llvm::SmallVector<unsigned> perm;
    for (int64_t i = 0; i < rank; ++i)
      perm.push_back(i);
    std::swap(perm[dim0], perm[dim1]);

    mlir::Type elementType = resultType.getElementType();

    // Output buffer has the result shape, not the input shape.
    mlir::tensor::EmptyOp empty = mlir::tensor::EmptyOp::create(
        rewriter, loc, resultType.getShape(), elementType);

    llvm::SmallVector<mlir::utils::IteratorType> iteratorTypes(
        rank, mlir::utils::IteratorType::parallel);

    // Loops walk the output: out[..d_i..] reads in[..d_perm[i]..].
    // A swap is its own inverse, so perm also works as the input map.
    mlir::AffineMap inputMap =
        mlir::AffineMap::getPermutationMap(perm, rewriter.getContext());
    mlir::AffineMap outputMap =
        mlir::AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());
    llvm::SmallVector<mlir::AffineMap> indexingMaps = {inputMap, outputMap};

    mlir::linalg::GenericOp genericOp = mlir::linalg::GenericOp::create(
        rewriter, loc, mlir::TypeRange{resultType},
        mlir::ValueRange{adaptor.getInput()},
        mlir::ValueRange{empty.getResult()}, indexingMaps, iteratorTypes,
        [&](mlir::OpBuilder &b, mlir::Location l, mlir::ValueRange args) {
          // Pure data movement: just forward the input element.
          mlir::linalg::YieldOp::create(b, l, args[0]);
        });

    rewriter.replaceOp(op, genericOp.getResult(0));
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

    target
        .addIllegalOp<alex::addLinalgOp, alex::addBCLinalgOp,
                      alex::reductionLinalgOp, alex::argminOp, alex::reluOp,
                      alex::transposeOp>();

    mlir::RewritePatternSet patterns(context);
    patterns.add<ConvertAccAddDivOp, ConvertReductionOp, ConvertaddBCLinalgOp,
                 ConvertArgminOp, ConvertreluOp, ConvertTransposeOp>(context);

    if (failed(mlir::applyPartialConversion(module, target,
                                            std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<mlir::Pass> createAlexToLinalgPass() {
  return std::make_unique<AlexToLinalgPass>();
}

void registerAlexToLinalgPass() {
  mlir::PassRegistration<AlexToLinalgPass>();
}
