#include "Alex/AlexOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include <functional>

namespace {
// Register patterns to lower Alex operations to Arith operations
template <typename AlexOp, typename FloatOp, typename IntOp,
          mlir::linalg::ElementwiseKind ElementwiseKind>
class ConvertBinaryOp : public mlir::OpConversionPattern<AlexOp> {
public:
  using Base = mlir::OpConversionPattern<AlexOp>;
  using OpAdaptor = typename Base::OpAdaptor;
  using Base::Base;

  // check whether the type is a floating point
  bool isFloat(mlir::Type type) const {
    return llvm::isa<mlir::FloatType>(type);
  }

  // check whether the type is an integer
  bool isInt(mlir::Type type) const {
    return llvm::isa<mlir::IntegerType>(type);
  }

  // check whether the type is a ranked tensor
  bool isTensor(mlir::Type type) const {
    return llvm::isa<mlir::RankedTensorType>(type);
  }

  // SCALAR: float op float
  mlir::LogicalResult
  lowerFloat(AlexOp op, OpAdaptor adaptor,
             mlir::ConversionPatternRewriter &rewriter) const {
    rewriter.replaceOpWithNewOp<FloatOp>(op, adaptor.getInput1(),
                                         adaptor.getInput2());

    return mlir::success();
  }

  // SCALAR: int op int
  mlir::LogicalResult
  lowerInt(AlexOp op, OpAdaptor adaptor,
           mlir::ConversionPatternRewriter &rewriter) const {
    rewriter.replaceOpWithNewOp<IntOp>(op, adaptor.getInput1(),
                                       adaptor.getInput2());

    return mlir::success();
  }

  // SCALAR: int op float
  // convert the integer input to the float
  mlir::LogicalResult
  lowerIntFloat(AlexOp op, OpAdaptor adaptor,
                mlir::ConversionPatternRewriter &rewriter) const {
    mlir::FloatType floatType =
        llvm::cast<mlir::FloatType>(op.getInput2().getType());

    mlir::arith::SIToFPOp convertedInt = mlir::arith::SIToFPOp::create(
        rewriter, op.getLoc(), floatType, adaptor.getInput1());

    rewriter.replaceOpWithNewOp<FloatOp>(op, convertedInt, adaptor.getInput2());

    return mlir::success();
  }

  // SCALAR: float op int
  // convert the integer input to the float
  mlir::LogicalResult
  lowerFloatInt(AlexOp op, OpAdaptor adaptor,
                mlir::ConversionPatternRewriter &rewriter) const {
    mlir::FloatType floatType =
        llvm::cast<mlir::FloatType>(op.getInput1().getType());

    mlir::arith::SIToFPOp convertedInt = mlir::arith::SIToFPOp::create(
        rewriter, op.getLoc(), floatType, adaptor.getInput2());

    rewriter.replaceOpWithNewOp<FloatOp>(op, adaptor.getInput1(), convertedInt);

    return mlir::success();
  }

  // Scalar + Tensor or Tensor + Scalar
  // convert the scalar into a tensor by filling an empty tensor with scalar
  // value, then perform the tensor operation
  mlir::LogicalResult
  lowerScalarTensor(AlexOp op, OpAdaptor adaptor,
                    mlir::ConversionPatternRewriter &rewriter) const {
    mlir::Type input1Type = op.getInput1().getType();
    mlir::Type input2Type = op.getInput2().getType();

    mlir::Value tensor;
    mlir::Value scalar;
    mlir::RankedTensorType tensorType;

    // Identify which operand is the tensor and which is the scalar.
    if (isTensor(input1Type)) {
      tensor = adaptor.getInput1();
      scalar = adaptor.getInput2();
      tensorType = llvm::cast<mlir::RankedTensorType>(input1Type);
    } else {
      scalar = adaptor.getInput1();
      tensor = adaptor.getInput2();
      tensorType = llvm::cast<mlir::RankedTensorType>(input2Type);
    }

    // Create an empty tensor with the same shape and element type as the tensor
    // operand.
    mlir::tensor::EmptyOp emptyTensor = mlir::tensor::EmptyOp::create(
        rewriter, op.getLoc(), tensorType.getShape(),
        tensorType.getElementType());
    // Fill the empty tensor with the scalar value.
    mlir::linalg::FillOp fillOp = mlir::linalg::FillOp::create(
        rewriter, op.getLoc(), scalar, emptyTensor.getResult());

    if (isTensor(input1Type)) {
      return lowerTensor(op, tensor, fillOp.getResult(0), tensorType, rewriter);
    }

    return lowerTensor(op, fillOp.getResult(0), tensor, tensorType, rewriter);
  }

  // Tensor operation
  // create a linalg.elementwise operation
  mlir::LogicalResult
  lowerTensor(AlexOp op, mlir::Value input1, mlir::Value input2,
              mlir::RankedTensorType tensorType,
              mlir::ConversionPatternRewriter &rewriter) const {
    // Fill the empty tensor with the scalar value.
    mlir::tensor::EmptyOp resultTensor = mlir::tensor::EmptyOp::create(
        rewriter, op.getLoc(), tensorType.getShape(),
        tensorType.getElementType());
    // Specify whether the elementwise operation is add, sub, or mul.
    mlir::linalg::ElementwiseKindAttr kindAttr =
        mlir::linalg::ElementwiseKindAttr::get(rewriter.getContext(),
                                               ElementwiseKind);
    mlir::AffineMap identityMap = mlir::AffineMap::getMultiDimIdentityMap(
        tensorType.getRank(), rewriter.getContext());
    mlir::ArrayAttr indexingMaps =
        rewriter.getAffineMapArrayAttr({identityMap, identityMap, identityMap});
    // Create the linalg.elementwise operation.
    mlir::linalg::ElementwiseOp elementwiseOp =
        mlir::linalg::ElementwiseOp::create(
            rewriter, op.getLoc(), mlir::ValueRange{input1, input2},
            mlir::ValueRange{resultTensor.getResult()}, kindAttr, indexingMaps);
    // Replace the original Alex operation with the newly created linalg
    // operation.
    rewriter.replaceOp(op, elementwiseOp.getResults());
    return mlir::success();
  }

  // tensor + tensor
  // Both tensor must have same type
  mlir::LogicalResult
  lowerTensorTensor(AlexOp op, OpAdaptor adaptor,
                    mlir::ConversionPatternRewriter &rewriter) const {
    mlir::RankedTensorType input1Type =
        llvm::cast<mlir::RankedTensorType>(op.getInput1().getType());
    mlir::RankedTensorType input2Type =
        llvm::cast<mlir::RankedTensorType>(op.getInput2().getType());
    // Elementwise operations require matching tensor types here.
    if (input1Type != input2Type)
      return mlir::failure();
    return lowerTensor(op, adaptor.getInput1(), adaptor.getInput2(), input1Type,
                       rewriter);
  }

  // Main rewrite function
  // Find the operand combination and call the appropriate lowering functions
  mlir::LogicalResult
  matchAndRewrite(AlexOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Type input1Type = op.getInput1().getType();

    mlir::Type input2Type = op.getInput2().getType();

    // float op float
    if (isFloat(input1Type) && isFloat(input2Type))
      return lowerFloat(op, adaptor, rewriter);

    // int op int
    if (isInt(input1Type) && isInt(input2Type))
      return lowerInt(op, adaptor, rewriter);

    // int op float
    if (isInt(input1Type) && isFloat(input2Type))
      return lowerIntFloat(op, adaptor, rewriter);

    // float op int
    if (isFloat(input1Type) && isInt(input2Type))
      return lowerFloatInt(op, adaptor, rewriter);

    // int or float op tensor
    if ((isInt(input1Type) || isFloat(input1Type)) && isTensor(input2Type))
      return lowerScalarTensor(op, adaptor, rewriter);

    // tensor op int or float
    if (isTensor(input1Type) && (isInt(input2Type) || isFloat(input2Type)))
      return lowerScalarTensor(op, adaptor, rewriter);

    // tensor op tensor
    if (isTensor(input1Type) && isTensor(input2Type))
      return lowerTensorTensor(op, adaptor, rewriter);

    return mlir::failure();
  }
};

// ADD: alex.add
using ConvertAddOp =
    ConvertBinaryOp<alex::AddOp, mlir::arith::AddFOp, mlir::arith::AddIOp,
                    mlir::linalg::ElementwiseKind::add>;
// SUB: alex.sun
using ConvertSubOp =
    ConvertBinaryOp<alex::SubOp, mlir::arith::SubFOp, mlir::arith::SubIOp,
                    mlir::linalg::ElementwiseKind::sub>;
// MUL: alex.mul
using ConvertMulOp =
    ConvertBinaryOp<alex::MulOp, mlir::arith::MulFOp, mlir::arith::MulIOp,
                    mlir::linalg::ElementwiseKind::mul>;

class ConvertConstOp : public mlir::OpConversionPattern<alex::ConstOp> {
public:
  using mlir::OpConversionPattern<alex::ConstOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(alex::ConstOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // Get the constant value
    mlir::TypedAttr value = llvm::dyn_cast<mlir::TypedAttr>(op.getValueAttr());

    // Fail if the value is not a typed attribute
    if (!value)
      return mlir::failure();

    // Lower alex.const to arith.constant.
    rewriter.replaceOpWithNewOp<mlir::arith::ConstantOp>(op, value);

    return mlir::success();
  }
};

// conversion pattern for addcmul operation
class ConvertAddcmulOp : public mlir::OpConversionPattern<alex::AddcmulOp> {
public:
  using mlir::OpConversionPattern<alex::AddcmulOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(alex::AddcmulOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::RankedTensorType input =
        llvm::cast<mlir::RankedTensorType>(op.getInput().getType());
    mlir::RankedTensorType tensor1 =
        llvm::cast<mlir::RankedTensorType>(op.getTensor1().getType());
    mlir::RankedTensorType tensor2 =
        llvm::cast<mlir::RankedTensorType>(op.getTensor2().getType());

    // Calculate the broadcasted shape of input and tensor1.
    llvm::SmallVector<int64_t> broadcastShape;

    if (!mlir::OpTrait::util::getBroadcastedShape(
            input.getShape(), tensor1.getShape(), broadcastShape)) {
      return rewriter.notifyMatchFailure(
          op, "input and tensor1 shapes are not broadcastable");
    }

    // Calculate the broadcasted shape with tensor2.
    llvm::SmallVector<int64_t> finalShape;

    if (!mlir::OpTrait::util::getBroadcastedShape(
            broadcastShape, tensor2.getShape(), finalShape)) {
      return rewriter.notifyMatchFailure(
          op, "input, tensor1 and tensor2 shapes are not broadcastable");
    }

    mlir::FloatAttr valueAttr = llvm::cast<mlir::FloatAttr>(op.getValueAttr());

    // Create an identity map for the final broadcasted shape.
    mlir::AffineMap identityMap = mlir::AffineMap::getMultiDimIdentityMap(
        finalShape.size(), rewriter.getContext());

    // Create an indexing map for a broadcasted operand.
    std::function<mlir::AffineMap(mlir::ArrayRef<int64_t>)> getBroadcastMap =
        [&](mlir::ArrayRef<int64_t> operandShape) -> mlir::AffineMap {
      unsigned finalRank = finalShape.size();
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

    mlir::AffineMap inputMap = getBroadcastMap(input.getShape());
    mlir::AffineMap tensor1Map = getBroadcastMap(tensor1.getShape());
    mlir::AffineMap tensor2Map = getBroadcastMap(tensor2.getShape());

    // Create an empty tensor with the final broadcasted shape.
    mlir::tensor::EmptyOp emptyTensor = mlir::tensor::EmptyOp::create(
        rewriter, op.getLoc(), finalShape, input.getElementType());

    // Convert scalar value into arith constant.
    mlir::arith::ConstantOp value =
        mlir::arith::ConstantOp::create(rewriter, op.getLoc(), valueAttr);

    // Fill the scalar value into a tensor.
    mlir::linalg::FillOp valueTensor = mlir::linalg::FillOp::create(
        rewriter, op.getLoc(), value.getResult(), emptyTensor.getResult());

    // tensor1 * tensor2
    mlir::tensor::EmptyOp emptyMulTensor = mlir::tensor::EmptyOp::create(
        rewriter, op.getLoc(), finalShape, input.getElementType());

    mlir::linalg::ElementwiseKindAttr mulKind =
        mlir::linalg::ElementwiseKindAttr::get(
            rewriter.getContext(), mlir::linalg::ElementwiseKind::mul);

    mlir::ArrayAttr mulIndexingMaps =
        rewriter.getAffineMapArrayAttr({tensor1Map, tensor2Map, identityMap});

    // Perform tensor1 * tensor2.
    mlir::linalg::ElementwiseOp mul1 = mlir::linalg::ElementwiseOp::create(
        rewriter, op.getLoc(),
        mlir::ValueRange{adaptor.getTensor1(), adaptor.getTensor2()},
        mlir::ValueRange{emptyMulTensor.getResult()}, mulKind, mulIndexingMaps);

    // value * (tensor1 * tensor2)
    mlir::tensor::EmptyOp emptyMul2Tensor = mlir::tensor::EmptyOp::create(
        rewriter, op.getLoc(), finalShape, input.getElementType());

    mlir::ArrayAttr mul2IndexingMaps =
        rewriter.getAffineMapArrayAttr({identityMap, identityMap, identityMap});

    mlir::linalg::ElementwiseOp mul2 = mlir::linalg::ElementwiseOp::create(
        rewriter, op.getLoc(),
        mlir::ValueRange{valueTensor.getResult(0), mul1.getResult(0)},
        mlir::ValueRange{emptyMul2Tensor.getResult()}, mulKind,
        mul2IndexingMaps);

    // input + (value * tensor1 * tensor2)
    mlir::tensor::EmptyOp emptyResultTensor = mlir::tensor::EmptyOp::create(
        rewriter, op.getLoc(), finalShape, input.getElementType());

    mlir::linalg::ElementwiseKindAttr addKind =
        mlir::linalg::ElementwiseKindAttr::get(
            rewriter.getContext(), mlir::linalg::ElementwiseKind::add);

    mlir::ArrayAttr addIndexingMaps =
        rewriter.getAffineMapArrayAttr({inputMap, identityMap, identityMap});

    mlir::linalg::ElementwiseOp add = mlir::linalg::ElementwiseOp::create(
        rewriter, op.getLoc(),
        mlir::ValueRange{adaptor.getInput(), mul2.getResult(0)},
        mlir::ValueRange{emptyResultTensor.getResult()}, addKind,
        addIndexingMaps);

    // Replace the original alex.addcmul operation.
    rewriter.replaceOp(op, add.getResults());

    return mlir::success();
  }
};

// Argmin Lowering:

// alex.argmin
//   -> if dim is not present
//      - find the minimum in the entire tensor
//      - get it flattened index
//      - store that index in the result tensor

//   -> if dim is present
//      - find the minimum along that dimension
//      - loop over the remaining dimensions
//      - get the index of the minimum
//      - store that index in the result tensor
//
// Keep the reduced dimesnion with size 1 id keepdim is true
// remove the reduced dimension if keepdim is false

// scf.for -> tensor.extract -> airth.cmpi/cmpf -> arith.select -> tensor.insert

class ConvertArgminOp : public mlir::OpConversionPattern<alex::ArgminOp> {
public:
  using mlir::OpConversionPattern<alex::ArgminOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(alex::ArgminOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();

    // Input tensor.
    mlir::RankedTensorType input =
        llvm::dyn_cast<mlir::RankedTensorType>(op.getInput().getType());

    if (!input)
      return rewriter.notifyMatchFailure(op, "input must be a ranked tensor");

    unsigned rank = input.getRank();

    // get the value of keepdim
    mlir::BoolAttr keepdimAttr = op->getAttrOfType<mlir::BoolAttr>("keepdim");

    bool keepdim = keepdimAttr && keepdimAttr.getValue();

    bool isFloat = llvm::isa<mlir::FloatType>(input.getElementType());

    // Common constants.
    mlir::Value zero = mlir::arith::ConstantIndexOp::create(rewriter, loc, 0);

    mlir::Value one = mlir::arith::ConstantIndexOp::create(rewriter, loc, 1);

    mlir::Value zeroI64 = mlir::arith::ConstantOp::create(
        rewriter, loc, rewriter.getI64IntegerAttr(0));

    // Get the runtime size of every input dimension.
    llvm::SmallVector<mlir::Value> inputDimSizes;

    for (unsigned i = 0; i < rank; ++i) {
      if (input.isDynamicDim(i)) {
        // Dynamic dimension = tensor.dim
        mlir::Value dimIndex =
            mlir::arith::ConstantIndexOp::create(rewriter, loc, i);

        mlir::Value size = mlir::tensor::DimOp::create(
            rewriter, loc, adaptor.getInput(), dimIndex);

        inputDimSizes.push_back(size);
      } else {
        // Static dimension = constant index
        inputDimSizes.push_back(mlir::arith::ConstantIndexOp::create(
            rewriter, loc, input.getDimSize(i)));
      }
    }

    // CASE 1:
    // dim is absent -> dim=None
    // Reduce over the entire flattened tensor.
    mlir::IntegerAttr dimAttr = op->getAttrOfType<mlir::IntegerAttr>("dim");

    if (!dimAttr) {
      // Check statically known zero dimensions.
      for (unsigned i = 0; i < rank; ++i) {
        if (!input.isDynamicDim(i) && input.getDimSize(i) == 0) {
          return rewriter.notifyMatchFailure(op,
                                             "cannot reduce an empty tensor");
        }
      }

      // Build the result based on keepdim
      llvm::SmallVector<int64_t> resultShape;

      if (keepdim)
        resultShape.assign(rank, 1);

      mlir::Value emptyResult = mlir::tensor::EmptyOp::create(
          rewriter, loc, resultShape, rewriter.getI64Type());

      // Initialize the input indices to zero
      llvm::SmallVector<mlir::Value> zeroIndices(rank, zero);

      mlir::Value firstValue = mlir::tensor::ExtractOp::create(
          rewriter, loc, adaptor.getInput(), zeroIndices);

      // Recursive nested loops over ALL input dimensions, carry the minValue
      // and minIndex
      std::function<llvm::SmallVector<mlir::Value, 2>(
          mlir::OpBuilder &, mlir::Location, unsigned, mlir::Value, mlir::Value,
          llvm::SmallVectorImpl<mlir::Value> &)>
          buildGlobalReduction;

      buildGlobalReduction = [&](mlir::OpBuilder &builder,
                                 mlir::Location bodyLoc, unsigned currentDim,
                                 mlir::Value minValue, mlir::Value minIndex,
                                 llvm::SmallVectorImpl<mlir::Value> &indices)
          -> llvm::SmallVector<mlir::Value, 2> {
        // All dimensions have been traversed.
        // Process this single element.
        if (currentDim == rank) {
          mlir::Value currentValue = mlir::tensor::ExtractOp::create(
              builder, bodyLoc, adaptor.getInput(), indices);

          // Calculate flattened index.
          // flatIndex * dimensionSize + index
          mlir::Value flatIndex = zeroI64;

          for (unsigned i = 0; i < rank; ++i) {
            mlir::Value indexI64 = mlir::arith::IndexCastOp::create(
                builder, bodyLoc, builder.getI64Type(), indices[i]);

            mlir::Value dimSizeI64 = mlir::arith::IndexCastOp::create(
                builder, bodyLoc, builder.getI64Type(), inputDimSizes[i]);

            mlir::Value multiplied = mlir::arith::MulIOp::create(
                builder, bodyLoc, flatIndex, dimSizeI64);

            flatIndex = mlir::arith::AddIOp::create(builder, bodyLoc,
                                                    multiplied, indexI64);
          }

          // Compare current value with current minimum.
          mlir::Value condition;

          if (isFloat) {
            condition = mlir::arith::CmpFOp::create(
                builder, bodyLoc, mlir::arith::CmpFPredicate::OLT, currentValue,
                minValue);
          } else {
            condition = mlir::arith::CmpIOp::create(
                builder, bodyLoc, mlir::arith::CmpIPredicate::slt, currentValue,
                minValue);
          }

          mlir::Value newMinValue = mlir::arith::SelectOp::create(
              builder, bodyLoc, condition, currentValue, minValue);

          mlir::Value newMinIndex = mlir::arith::SelectOp::create(
              builder, bodyLoc, condition, flatIndex, minIndex);

          return {newMinValue, newMinIndex};
        }

        // Loop over current dimension.
        mlir::scf::ForOp loop = mlir::scf::ForOp::create(
            builder, bodyLoc, zero, inputDimSizes[currentDim], one,
            mlir::ValueRange{minValue, minIndex},
            [&](mlir::OpBuilder &nestedBuilder, mlir::Location nestedLoc,
                mlir::Value inductionVariable, mlir::ValueRange iterArgs) {
              indices.push_back(inductionVariable);

              llvm::SmallVector<mlir::Value, 2> reductionResult =
                  buildGlobalReduction(nestedBuilder, nestedLoc, currentDim + 1,
                                       iterArgs[0], iterArgs[1], indices);

              indices.pop_back();

              mlir::scf::YieldOp::create(nestedBuilder, nestedLoc,
                                         reductionResult);
            });

        return {loop.getResult(0), loop.getResult(1)};
      };

      llvm::SmallVector<mlir::Value> indices;

      llvm::SmallVector<mlir::Value, 2> finalReduction =
          buildGlobalReduction(rewriter, loc, 0, firstValue, zeroI64, indices);

      // Insert final flattened index into result tensor.
      llvm::SmallVector<mlir::Value> resultIndices;

      if (keepdim)
        resultIndices.assign(rank, zero);

      mlir::Value finalResult = mlir::tensor::InsertOp::create(
          rewriter, loc, finalReduction[1], emptyResult, resultIndices);

      rewriter.replaceOp(op, finalResult);

      return mlir::success();
    }

    // CASE 2:
    // dim is specified.
    int64_t dim = dimAttr.getInt();

    // Normalize negative dimension.
    if (dim < 0)
      dim += rank;

    if (dim < 0 || dim >= static_cast<int64_t>(rank))
      return rewriter.notifyMatchFailure(op, "dimension is out of range");

    unsigned reductionDim = static_cast<unsigned>(dim);

    // Determine result shape.
    // One output dimension exists for every non-reduced dimension.
    llvm::SmallVector<int64_t> resultShape;
    llvm::SmallVector<mlir::Value> dynamicSizes;
    llvm::SmallVector<mlir::Value> outputBounds;

    for (unsigned i = 0; i < rank; ++i) {
      if (i == reductionDim) {
        if (keepdim)
          resultShape.push_back(1);

        continue;
      }

      if (input.isDynamicDim(i)) {
        resultShape.push_back(mlir::ShapedType::kDynamic);

        dynamicSizes.push_back(inputDimSizes[i]);

        outputBounds.push_back(inputDimSizes[i]);
      } else {
        resultShape.push_back(input.getDimSize(i));

        outputBounds.push_back(inputDimSizes[i]);
      }
    }

    // Reduction dimension size.
    mlir::Value reductionBound = inputDimSizes[reductionDim];

    if (!input.isDynamicDim(reductionDim) &&
        input.getDimSize(reductionDim) == 0) {
      return rewriter.notifyMatchFailure(op,
                                         "cannot reduce an empty dimension");
    }

    // Result tensor.
    mlir::Value emptyResult = mlir::tensor::EmptyOp::create(
        rewriter, loc, resultShape, rewriter.getI64Type(), dynamicSizes);

    // Insert the reduction index into the non-reduced indices.
    std::function<llvm::SmallVector<mlir::Value>(llvm::ArrayRef<mlir::Value>,
                                                 mlir::Value)>
        insertAtReductionDim;
    insertAtReductionDim =
        [&](llvm::ArrayRef<mlir::Value> keptIndices,
            mlir::Value reductionIndex) -> llvm::SmallVector<mlir::Value> {
      llvm::SmallVector<mlir::Value> fullIndices;

      fullIndices.reserve(rank);

      unsigned keptPosition = 0;

      for (unsigned i = 0; i < rank; ++i) {
        if (i == reductionDim) {
          fullIndices.push_back(reductionIndex);
          continue;
        }

        fullIndices.push_back(keptIndices[keptPosition]);

        ++keptPosition;
      }

      return fullIndices;
    };

    // Recursive loops over all non-reduced dimensions.
    std::function<mlir::Value(mlir::OpBuilder &, mlir::Location, unsigned,
                              mlir::Value,
                              llvm::SmallVectorImpl<mlir::Value> &)>
        buildOutputLoops;

    buildOutputLoops =
        [&](mlir::OpBuilder &builder, mlir::Location bodyLoc,
            unsigned outputDim, mlir::Value currentResult,
            llvm::SmallVectorImpl<mlir::Value> &outputIndices) -> mlir::Value {
      // All non-reduced dimensions have been generated.
      // Perform the reduction along `dim`.
      if (outputDim == outputBounds.size()) {

        // Start with first element along reduction dimension.
        llvm::SmallVector<mlir::Value> firstInputIndices =
            insertAtReductionDim(outputIndices, zero);

        mlir::Value firstValue = mlir::tensor::ExtractOp::create(
            builder, bodyLoc, adaptor.getInput(), firstInputIndices);

        // First element has index 0.
        mlir::scf::ForOp reductionLoop = mlir::scf::ForOp::create(
            builder, bodyLoc, one, reductionBound, one,
            mlir::ValueRange{firstValue, zeroI64},
            [&](mlir::OpBuilder &innerBuilder, mlir::Location innerLoc,
                mlir::Value reductionIndex, mlir::ValueRange reductionArgs) {
              mlir::Value minValue = reductionArgs[0];

              mlir::Value minIndex = reductionArgs[1];

              // Read current element.
              llvm::SmallVector<mlir::Value> currentInputIndices =
                  insertAtReductionDim(outputIndices, reductionIndex);

              mlir::Value currentValue = mlir::tensor::ExtractOp::create(
                  innerBuilder, innerLoc, adaptor.getInput(),
                  currentInputIndices);

              // Compare.
              mlir::Value condition;

              if (isFloat) {
                condition = mlir::arith::CmpFOp::create(
                    innerBuilder, innerLoc, mlir::arith::CmpFPredicate::OLT,
                    currentValue, minValue);
              } else {
                condition = mlir::arith::CmpIOp::create(
                    innerBuilder, innerLoc, mlir::arith::CmpIPredicate::slt,
                    currentValue, minValue);
              }

              // Select new minimum value.
              mlir::Value newMinValue = mlir::arith::SelectOp::create(
                  innerBuilder, innerLoc, condition, currentValue, minValue);

              // Convert reduction index to i64.
              mlir::Value reductionIndexI64 = mlir::arith::IndexCastOp::create(
                  innerBuilder, innerLoc, innerBuilder.getI64Type(),
                  reductionIndex);

              // Select new minimum index.
              mlir::Value newMinIndex = mlir::arith::SelectOp::create(
                  innerBuilder, innerLoc, condition, reductionIndexI64,
                  minIndex);

              // Carry min value and min index.
              mlir::scf::YieldOp::create(
                  innerBuilder, innerLoc,
                  mlir::ValueRange{newMinValue, newMinIndex});
            });

        mlir::Value resultIndex = reductionLoop.getResult(1);

        // Output indices.
        llvm::SmallVector<mlir::Value> resultIndices;

        if (keepdim) {
          // Insert reduction dimension with index 0.
          resultIndices = insertAtReductionDim(outputIndices, zero);
        } else {
          resultIndices.assign(outputIndices.begin(), outputIndices.end());
        }

        // Insert argmin index into result.
        return mlir::tensor::InsertOp::create(builder, bodyLoc, resultIndex,
                                              currentResult, resultIndices)
            .getResult();
      }

      // Loop over one non-reduced dimension.
      mlir::Value upperBound = outputBounds[outputDim];

      mlir::scf::ForOp loop = mlir::scf::ForOp::create(
          builder, bodyLoc, zero, upperBound, one,
          mlir::ValueRange{currentResult},
          [&](mlir::OpBuilder &nestedBuilder, mlir::Location nestedLoc,
              mlir::Value inductionVariable, mlir::ValueRange iterArgs) {
            outputIndices.push_back(inductionVariable);

            mlir::Value updatedResult =
                buildOutputLoops(nestedBuilder, nestedLoc, outputDim + 1,
                                 iterArgs[0], outputIndices);

            outputIndices.pop_back();

            mlir::scf::YieldOp::create(nestedBuilder, nestedLoc, updatedResult);
          });

      return loop.getResult(0);
    };

    // Generate output loops.
    llvm::SmallVector<mlir::Value> outputIndices;

    mlir::Value finalResult =
        buildOutputLoops(rewriter, loc, 0, emptyResult, outputIndices);

    rewriter.replaceOp(op, finalResult);

    return mlir::success();
  }
};

class AlexToArithPass
    : public mlir::PassWrapper<AlexToArithPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AlexToArithPass)

  // PASS name
  mlir::StringRef getArgument() const final { return "convert-alex-to-arith"; }

  // Pass Description
  mlir::StringRef getDescription() const final {
    return "Lower Alex Operations to Arith operation";
  }

  //  Register dialects used by the lowering.
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithDialect, mlir::linalg::LinalgDialect,
                    mlir::tensor::TensorDialect, mlir::scf::SCFDialect>();
  }

  // entry point of lowering pass
  void runOnOperation() override {
    mlir::MLIRContext &context = getContext();
    mlir::ModuleOp module = getOperation();

    // register leagal and illegal operations fro conversion
    mlir::ConversionTarget target(context);

    // These dialects are allowed to remain after the conversion.
    target
        .addLegalDialect<mlir::arith::ArithDialect, mlir::linalg::LinalgDialect,
                         mlir::tensor::TensorDialect, mlir::scf::SCFDialect>();
    // Alex operations must be lowered
    target.addIllegalOp<alex::AddOp, alex::SubOp, alex::MulOp, alex::AddcmulOp,
                        alex::ArgminOp>();

    mlir::RewritePatternSet patterns(&context);

    // Register patterns that perform the actual lowering.
    patterns.add<ConvertAddOp, ConvertConstOp, ConvertSubOp, ConvertMulOp,
                 ConvertAddcmulOp, ConvertArgminOp>(&context);

    // Apply the conversion and fail the pass if any illegal. Alex operation
    // could not be lowered.
    if (failed(mlir::applyPartialConversion(module, target,
                                            std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

// Create and return the Alex-to-Arith lowering pass.
std::unique_ptr<mlir::Pass> createAlexToArithPass() {
  return std::make_unique<AlexToArithPass>();
}

// Register the pass with MLIR.
void registerAlexToArithPass() { mlir::PassRegistration<AlexToArithPass>(); }