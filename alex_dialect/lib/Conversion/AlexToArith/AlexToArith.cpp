#include "Alex/AlexOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
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

// conversion pattern for arange operation
// .arange(start = 0, end, step = 1)
// Lowers alex.arange into standard ops (arith/math/scf/tensor)
// size = max(0, ceil((end - start) / step))
// out[i] = start + i * step for i in [0, size)
class ConvertarangeOp : public mlir::OpConversionPattern<alex::RangeOp> {
public:
  using mlir::OpConversionPattern<alex::RangeOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(alex::RangeOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {

    mlir::Location loc = op.getLoc();

    // end is runtime value (operand)
    mlir::Value end = op.getEnd();

    // Check whether any input is float.
    bool hasFloat = llvm::isa<mlir::FloatAttr>(op.getStartAttr()) ||
                    llvm::isa<mlir::FloatType>(end.getType()) ||
                    llvm::isa<mlir::FloatAttr>(op.getStepAttr());

    mlir::Type elementType;

    // If any of the inputs is float, output tensor is f32 or f64
    // if all 3 inputs are int -> output tensor is i32 or i64
    if (hasFloat) {
      // If any input is explicitly f64, use f64 else use f32, same for int
      bool useF64 =
          (llvm::isa<mlir::FloatAttr>(op.getStartAttr()) &&
           llvm::cast<mlir::FloatAttr>(op.getStartAttr()).getType().isF64()) ||
          (llvm::isa<mlir::FloatAttr>(op.getStepAttr()) &&
           llvm::cast<mlir::FloatAttr>(op.getStepAttr()).getType().isF64()) ||
          (llvm::isa<mlir::FloatType>(end.getType()) &&
           llvm::cast<mlir::FloatType>(end.getType()).isF64());

      elementType = useF64 ? rewriter.getF64Type() : rewriter.getF32Type();
    } else {
      bool useI64 =
          (llvm::isa<mlir::IntegerAttr>(op.getStartAttr()) &&
           llvm::cast<mlir::IntegerAttr>(op.getStartAttr())
               .getType()
               .isInteger(64)) ||
          (llvm::isa<mlir::IntegerAttr>(op.getStepAttr()) &&
           llvm::cast<mlir::IntegerAttr>(op.getStepAttr())
               .getType()
               .isInteger(64)) ||
          (llvm::isa<mlir::IntegerType>(end.getType()) &&
           llvm::cast<mlir::IntegerType>(end.getType()).isInteger(64));

      elementType = useI64 ? rewriter.getI64Type() : rewriter.getI32Type();
    }

    // Create start constant, convert it  to the chosen element type
    mlir::TypedAttr newStartAttr;

    // check if the start attribute is an integer attribute
    if (llvm::isa<mlir::IntegerAttr>(op.getStartAttr())) {
      int64_t value = llvm::cast<mlir::IntegerAttr>(op.getStartAttr()).getInt();

      // according the chosen element type, convert it accordingly
      // for f32 and f64
      if (llvm::isa<mlir::FloatType>(elementType)) {
        newStartAttr =
            rewriter.getFloatAttr(elementType, static_cast<double>(value));
      }
      // for i32 and i64
      else {
        newStartAttr = rewriter.getIntegerAttr(elementType, value);
      }
    }
    // if start is float attribute
    else {
      double value =
          llvm::cast<mlir::FloatAttr>(op.getStartAttr()).getValueAsDouble();

      newStartAttr = rewriter.getFloatAttr(elementType, value);
    }

    mlir::Value start =
        mlir::arith::ConstantOp::create(rewriter, loc, newStartAttr);

    // Create step constant. [same operation as start]
    mlir::TypedAttr newStepAttr;

    if (llvm::isa<mlir::IntegerAttr>(op.getStepAttr())) {
      int64_t value = llvm::cast<mlir::IntegerAttr>(op.getStepAttr()).getInt();

      if (llvm::isa<mlir::FloatType>(elementType)) {
        newStepAttr =
            rewriter.getFloatAttr(elementType, static_cast<double>(value));
      } else {
        newStepAttr = rewriter.getIntegerAttr(elementType, value);
      }
    } else {
      double value =
          llvm::cast<mlir::FloatAttr>(op.getStepAttr()).getValueAsDouble();

      newStepAttr = rewriter.getFloatAttr(elementType, value);
    }

    mlir::Value step =
        mlir::arith::ConstantOp::create(rewriter, loc, newStepAttr);

    // Convert end to the selected element type, so that start, end and step are
    // all the same type.
    if (end.getType() != elementType) {

      // int -> float
      if (llvm::isa<mlir::IntegerType>(end.getType()) &&
          llvm::isa<mlir::FloatType>(elementType)) {

        end = mlir::arith::SIToFPOp::create(rewriter, loc, elementType, end);
      }
      // float -> float:: widen(f32 -> f64) or narrow(f64 -> f32)
      else if (llvm::isa<mlir::FloatType>(end.getType()) &&
               llvm::isa<mlir::FloatType>(elementType)) {

        mlir::FloatType sourceType = llvm::cast<mlir::FloatType>(end.getType());

        mlir::FloatType targetType = llvm::cast<mlir::FloatType>(elementType);

        if (sourceType.getWidth() < targetType.getWidth()) {
          end = mlir::arith::ExtFOp::create(rewriter, loc, elementType, end);
        } else {
          end = mlir::arith::TruncFOp::create(rewriter, loc, elementType, end);
        }
      }
      // int -> int: widen (i32 -> i64) or narrow (i64 -> i32)
      else if (llvm::isa<mlir::IntegerType>(end.getType()) &&
               llvm::isa<mlir::IntegerType>(elementType)) {

        mlir::IntegerType sourceType =
            llvm::cast<mlir::IntegerType>(end.getType());

        mlir::IntegerType targetType =
            llvm::cast<mlir::IntegerType>(elementType);

        if (sourceType.getWidth() < targetType.getWidth()) {
          end = mlir::arith::ExtSIOp::create(rewriter, loc, elementType, end);
        } else {
          end = mlir::arith::TruncIOp::create(rewriter, loc, elementType, end);
        }
      }
    }

    // end - start
    mlir::Value difference;

    if (llvm::isa<mlir::FloatType>(elementType)) {
      difference = mlir::arith::SubFOp::create(rewriter, loc, end, start);
    } else {
      difference = mlir::arith::SubIOp::create(rewriter, loc, end, start);
    }

    // Calculate number of elements:
    // ceil((end - start) / step)
    mlir::Value sizeValue;

    // temporarily make step and difference f64, to avoid float error build-up
    // in repeated addition
    {
      mlir::FloatType sizeType = rewriter.getF64Type();

      mlir::Value differenceF64;
      mlir::Value stepF64;

      if (llvm::isa<mlir::FloatType>(elementType)) {
        if (elementType != sizeType) {
          differenceF64 =
              mlir::arith::ExtFOp::create(rewriter, loc, sizeType, difference);

          stepF64 = mlir::arith::ExtFOp::create(rewriter, loc, sizeType, step);
        } else {
          differenceF64 = difference;
          stepF64 = step;
        }
      } else {
        differenceF64 =
            mlir::arith::SIToFPOp::create(rewriter, loc, sizeType, difference);

        stepF64 = mlir::arith::SIToFPOp::create(rewriter, loc, sizeType, step);
      }

      mlir::Value quotient =
          mlir::arith::DivFOp::create(rewriter, loc, differenceF64, stepF64);

      mlir::Value ceilValue =
          mlir::math::CeilOp::create(rewriter, loc, quotient);

      mlir::Value zeroFloat = mlir::arith::ConstantOp::create(
          rewriter, loc, rewriter.getF64FloatAttr(0.0));

      // max(size, 0): a step pointing away from end gives an empty tensor.
      mlir::Value sizeFloat =
          mlir::arith::MaximumFOp::create(rewriter, loc, ceilValue, zeroFloat);

      // f64 -> i64 -> index (tensor sizes and loop bounds must be `index`).
      mlir::Value sizeI64 = mlir::arith::FPToSIOp::create(
          rewriter, loc, rewriter.getI64Type(), sizeFloat);

      sizeValue = mlir::arith::IndexCastOp::create(
          rewriter, loc, rewriter.getIndexType(), sizeI64);
    }

    // Create tensor<? x elementType>.
    // The size is only known at runtime, hence the dynamic dimension
    mlir::RankedTensorType resultType =
        mlir::RankedTensorType::get({mlir::ShapedType::kDynamic}, elementType);

    // Uninitialized tensor: the loop below fills every element
    mlir::tensor::EmptyOp emptyTensor = mlir::tensor::EmptyOp::create(
        rewriter, loc, resultType, mlir::ValueRange{sizeValue});

    // for i = 0; i < size; i++
    mlir::Value zero = mlir::arith::ConstantIndexOp::create(rewriter, loc, 0);

    mlir::Value one = mlir::arith::ConstantIndexOp::create(rewriter, loc, 1);

    // The tensor is carried through the loop as an ietr_Arg, because
    // tensor.insert returns a new tensor instead od modifying in place
    mlir::scf::ForOp loop =
        mlir::scf::ForOp::create(rewriter, loc, zero, sizeValue, one,
                                 mlir::ValueRange{emptyTensor.getResult()});

    {
      mlir::OpBuilder::InsertionGuard guard(rewriter);

      rewriter.setInsertionPointToStart(loop.getBody());

      mlir::Value iv = loop.getInductionVar();

      // Convert i from index to element type.
      mlir::Value indexValue;

      if (llvm::isa<mlir::FloatType>(elementType)) {

        mlir::Value indexI64 = mlir::arith::IndexCastOp::create(
            rewriter, loc, rewriter.getI64Type(), iv);

        indexValue =
            mlir::arith::SIToFPOp::create(rewriter, loc, elementType, indexI64);
      } else {

        indexValue =
            mlir::arith::IndexCastOp::create(rewriter, loc, elementType, iv);
      }

      // i * step
      mlir::Value multiplied;

      if (llvm::isa<mlir::FloatType>(elementType)) {
        multiplied =
            mlir::arith::MulFOp::create(rewriter, loc, indexValue, step);
      } else {
        multiplied =
            mlir::arith::MulIOp::create(rewriter, loc, indexValue, step);
      }

      // start + i * step
      // multiplying avoids the error build-up of repeated float addition
      mlir::Value value;

      if (llvm::isa<mlir::FloatType>(elementType)) {
        value = mlir::arith::AddFOp::create(rewriter, loc, start, multiplied);
      } else {
        value = mlir::arith::AddIOp::create(rewriter, loc, start, multiplied);
      }

      // Insert value into tensor at posiiton i
      mlir::Value inserted = mlir::tensor::InsertOp::create(
          rewriter, loc, value, loop.getRegionIterArgs()[0],
          mlir::ValueRange{iv});

      // pass the updated tensor to the next iteration
      mlir::scf::YieldOp::create(rewriter, loc, inserted);
    }

    // Replace alex.arange with the tensor produced by the loop
    rewriter.replaceOp(op, loop.getResult(0));

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
                    mlir::tensor::TensorDialect, mlir::scf::SCFDialect,
                    mlir::math::MathDialect>();
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
                         mlir::tensor::TensorDialect, mlir::scf::SCFDialect,
                         mlir::math::MathDialect>();
    // Alex operations must be lowered
    target.addIllegalOp<alex::AddOp, alex::SubOp, alex::MulOp, alex::AddcmulOp,
                        alex::RangeOp>();

    mlir::RewritePatternSet patterns(&context);

    // Register patterns that perform the actual lowering.
    patterns.add<ConvertAddOp, ConvertConstOp, ConvertSubOp, ConvertMulOp,
                 ConvertAddcmulOp, ConvertarangeOp>(&context);

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