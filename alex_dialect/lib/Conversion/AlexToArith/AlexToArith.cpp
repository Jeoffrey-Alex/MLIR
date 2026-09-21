#include "Alex/AlexOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace
{
    // Register patterns to lower Alex operations to Arith operations
    template <typename AlexOp, typename FloatOp, typename IntOp, mlir::linalg::ElementwiseKind ElementwiseKind>
    class ConvertBinaryOp : public mlir::OpConversionPattern<AlexOp>
    {
    public:
        using Base = mlir::OpConversionPattern<AlexOp>;
        using OpAdaptor = typename Base::OpAdaptor;
        using Base::Base;

        //check whether the type is a floating point 
        bool isFloat(mlir::Type type) const
        {
            return llvm::isa<mlir::FloatType>(type);
        }

        //check whether the type is an integer
        bool isInt(mlir::Type type) const
        {
            return llvm::isa<mlir::IntegerType>(type);
        }

        //check whether the type is a ranked tensor
        bool isTensor(mlir::Type type) const
        {
            return llvm::isa<mlir::RankedTensorType>(type);
        }

        // SCALAR: float op float
        mlir::LogicalResult lowerFloat(AlexOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter) const
        {
            rewriter.replaceOpWithNewOp<FloatOp>(op, adaptor.getInput1(), adaptor.getInput2());

            return mlir::success();
        }

        // SCALAR: int op int
        mlir::LogicalResult lowerInt(AlexOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter) const
        {
            rewriter.replaceOpWithNewOp<IntOp>(op, adaptor.getInput1(), adaptor.getInput2());

            return mlir::success();
        }

        //SCALAR: int op float
        //convert the integer input to the float 
        mlir::LogicalResult lowerIntFloat(AlexOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter) const
        {
            mlir::FloatType floatType = llvm::cast<mlir::FloatType>(op.getInput2().getType());

            mlir::arith::SIToFPOp convertedInt = mlir::arith::SIToFPOp::create(rewriter, op.getLoc(), floatType, adaptor.getInput1());

            rewriter.replaceOpWithNewOp<FloatOp>(op, convertedInt, adaptor.getInput2());

            return mlir::success();
        }

        //SCALAR: float op int
        //convert the integer input to the float 
        mlir::LogicalResult lowerFloatInt(AlexOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter) const
        {
            mlir::FloatType floatType = llvm::cast<mlir::FloatType>(op.getInput1().getType());

            mlir::arith::SIToFPOp convertedInt = mlir::arith::SIToFPOp::create(rewriter, op.getLoc(), floatType, adaptor.getInput2());

            rewriter.replaceOpWithNewOp<FloatOp>(op, adaptor.getInput1(), convertedInt);

            return mlir::success();
        }
        
        //Scalar + Tensor or Tensor + Scalar
        //convert the scalar into a tensor by filling an empty tensor with scalar value, then perform the tensor operation
        mlir::LogicalResult lowerScalarTensor(AlexOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter) const
        {
            mlir::Type input1Type = op.getInput1().getType();
            mlir::Type input2Type = op.getInput2().getType();

            mlir::Value tensor;
            mlir::Value scalar;
            mlir::RankedTensorType tensorType;

            // Identify which operand is the tensor and which is the scalar.
            if (isTensor(input1Type))
            {
                tensor = adaptor.getInput1();
                scalar = adaptor.getInput2();
                tensorType = llvm::cast<mlir::RankedTensorType>(input1Type);
            }
            else
            {
                scalar = adaptor.getInput1();
                tensor = adaptor.getInput2();
                tensorType = llvm::cast<mlir::RankedTensorType>(input2Type);
            }

            
            // Create an empty tensor with the same shape and element type as the tensor operand.
            mlir::tensor::EmptyOp emptyTensor = mlir::tensor::EmptyOp::create(rewriter, op.getLoc(), tensorType.getShape(), tensorType.getElementType());
            // Fill the empty tensor with the scalar value.
            mlir::linalg::FillOp fillOp = mlir::linalg::FillOp::create(rewriter, op.getLoc(), scalar, emptyTensor.getResult());

            if (isTensor(input1Type))
            {
                return lowerTensor(op, tensor, fillOp.getResult(0), tensorType, rewriter);
            }

            return lowerTensor(op, fillOp.getResult(0), tensor, tensorType, rewriter);
        }
        
        //Tensor operation
        //create a linalg.elementwise operation 
        mlir::LogicalResult lowerTensor(AlexOp op, mlir::Value input1, mlir::Value input2, mlir::RankedTensorType tensorType, mlir::ConversionPatternRewriter &rewriter) const
        {
            // Fill the empty tensor with the scalar value.
            mlir::tensor::EmptyOp resultTensor = mlir::tensor::EmptyOp::create(rewriter, op.getLoc(), tensorType.getShape(), tensorType.getElementType());
            // Specify whether the elementwise operation is add, sub, or mul.
            mlir::linalg::ElementwiseKindAttr kindAttr = mlir::linalg::ElementwiseKindAttr::get(rewriter.getContext(), ElementwiseKind);
            mlir::AffineMap identityMap = mlir::AffineMap::getMultiDimIdentityMap(tensorType.getRank(), rewriter.getContext());
            mlir::ArrayAttr indexingMaps = rewriter.getAffineMapArrayAttr({identityMap, identityMap, identityMap});
            // Create the linalg.elementwise operation.
            mlir::linalg::ElementwiseOp elementwiseOp = mlir::linalg::ElementwiseOp::create(rewriter, op.getLoc(), mlir::ValueRange{input1, input2}, mlir::ValueRange{resultTensor.getResult()}, kindAttr, indexingMaps);
            // Replace the original Alex operation with the newly created linalg operation.
            rewriter.replaceOp(op, elementwiseOp.getResults());
            return mlir::success();
        }

        // tensor + tensor
        //Both tensor must have same type
        mlir::LogicalResult lowerTensorTensor(AlexOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter) const
        {
            mlir::RankedTensorType input1Type = llvm::cast<mlir::RankedTensorType>(op.getInput1().getType());
            mlir::RankedTensorType input2Type = llvm::cast<mlir::RankedTensorType>(op.getInput2().getType());
            // Elementwise operations require matching tensor types here.
            if (input1Type != input2Type)
                return mlir::failure();
            return lowerTensor(op, adaptor.getInput1(), adaptor.getInput2(), input1Type, rewriter);
        }

        // Main rewrite function
        // Find the operand combination and call the appropriate lowering functions
        mlir::LogicalResult matchAndRewrite(AlexOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter) const override
        {
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

    //ADD: alex.add
    using ConvertAddOp = ConvertBinaryOp<alex::AddOp, mlir::arith::AddFOp, mlir::arith::AddIOp, mlir::linalg::ElementwiseKind::add>;
    //SUB: alex.sun
    using ConvertSubOp = ConvertBinaryOp<alex::SubOp, mlir::arith::SubFOp, mlir::arith::SubIOp, mlir::linalg::ElementwiseKind::sub>;
    //MUL: alex.mul
    using ConvertMulOp = ConvertBinaryOp<alex::MulOp, mlir::arith::MulFOp, mlir::arith::MulIOp, mlir::linalg::ElementwiseKind::mul>;

    class ConvertAddcmulOp : public mlir::OpConversionPattern<alex::AddcmulOp>
    {
    public:
        using mlir::OpConversionPattern<alex::AddcmulOp>::OpConversionPattern;

        mlir::LogicalResult matchAndRewrite(alex::AddcmulOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter) const override
        {
            auto input = llvm::cast<mlir::RankedTensorType>(op.getInput().getType());
            auto tensor1 = llvm::cast<mlir::RankedTensorType>(op.getTensor1().getType());
            auto tensor2 = llvm::cast<mlir::RankedTensorType>(op.getTensor2().getType());
            auto valueAttr = llvm::cast<mlir::FloatAttr>(op.getValueAttr());

            auto emptyTensor = mlir::tensor::EmptyOp::create(rewriter, op.getLoc(), input.getShape(), input.getElementType());
            auto value = mlir::arith::ConstantOp::create(rewriter, op.getLoc(), valueAttr);
            auto valueTensor = mlir::linalg::FillOp::create(rewriter, op.getLoc(), value.getResult(), emptyTensor.getResult());

            // tensor1*tensor2
            auto emptyMulTensor = mlir::tensor::EmptyOp::create(rewriter, op.getLoc(), input.getShape(), input.getElementType());
            auto mulKind = mlir::linalg::ElementwiseKindAttr::get(rewriter.getContext(), mlir::linalg::ElementwiseKind::mul);
            auto identityMap = mlir::AffineMap::getMultiDimIdentityMap(input.getRank(), rewriter.getContext());
            auto indexingMaps = rewriter.getAffineMapArrayAttr({identityMap, identityMap, identityMap});
            auto mul1 = mlir::linalg::ElementwiseOp::create(rewriter, op.getLoc(), mlir::ValueRange{adaptor.getTensor1(), adaptor.getTensor2()}, mlir::ValueRange{emptyMulTensor.getResult()}, mulKind, indexingMaps);

            // valueTensor *  mul1
            auto emptyMul2Tensor = mlir::tensor::EmptyOp::create(rewriter, op.getLoc(), input.getShape(), input.getElementType());
            auto mul2 = mlir::linalg::ElementwiseOp::create(rewriter, op.getLoc(), mlir::ValueRange{valueTensor.getResult(0), mul1.getResult(0)}, mlir::ValueRange{emptyMul2Tensor.getResult()}, mulKind, indexingMaps);
            auto emptyResultTensor = mlir::tensor::EmptyOp::create(rewriter, op.getLoc(), input.getShape(), input.getElementType());

            //add+mul2
            auto addKind = mlir::linalg::ElementwiseKindAttr::get(rewriter.getContext(), mlir::linalg::ElementwiseKind::add);
            auto add = mlir::linalg::ElementwiseOp::create(rewriter, op.getLoc(), mlir::ValueRange{adaptor.getInput(), mul2.getResult(0)}, mlir::ValueRange{emptyResultTensor.getResult()}, addKind, indexingMaps);

            rewriter.replaceOp(op, add.getResults());

            return mlir::success();
        }
    };

    class ConvertConstOp : public mlir::OpConversionPattern<alex::ConstOp>
    {
    public:
        using mlir::OpConversionPattern<alex::ConstOp>::OpConversionPattern;

        mlir::LogicalResult matchAndRewrite(alex::ConstOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter) const override
        {
            //Get the constant value
            mlir::TypedAttr value = llvm::dyn_cast<mlir::TypedAttr>(op.getValueAttr());

            //Fail if the value is not a typed attribute
            if (!value)
                return mlir::failure();

            // Lower alex.const to arith.constant.
            rewriter.replaceOpWithNewOp<mlir::arith::ConstantOp>(op, value);

            return mlir::success();
        }
    };

    class AlexToArithPass : public mlir::PassWrapper<AlexToArithPass, mlir::OperationPass<mlir::ModuleOp>>
    {
    public:
        MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AlexToArithPass)

        // PASS name
        mlir::StringRef getArgument() const final
        {
            return "convert-alex-to-arith";
        }

        // Pass Description
        mlir::StringRef getDescription() const final
        {
            return "Lower Alex Operations to Arith operation";
        }

        //  Register dialects used by the lowering.
        void getDependentDialects(mlir::DialectRegistry &registry) const override
        {
            registry.insert<
                mlir::arith::ArithDialect,
                mlir::linalg::LinalgDialect,
                mlir::tensor::TensorDialect>();
        }

        // entry point of lowering pass
        void runOnOperation() override
        {
            mlir::MLIRContext &context = getContext();
            mlir::ModuleOp module = getOperation();

            // register leagal and illegal operations fro conversion
            mlir::ConversionTarget target(context);

            //These dialects are allowed to remain after the conversion.
            target.addLegalDialect<mlir::arith::ArithDialect, mlir::linalg::LinalgDialect, mlir::tensor::TensorDialect>();
            //Alex operations must be lowered
            target.addIllegalOp<alex::AddOp, alex::SubOp, alex::MulOp>();
            target.addIllegalOp<alex::AddcmulOp>();

            mlir::RewritePatternSet patterns(&context);

            // Register patterns that perform the actual lowering.
            patterns.add<ConvertAddOp, ConvertConstOp, ConvertSubOp, ConvertMulOp, ConvertAddcmulOp>(&context);

            //Apply the conversion and fail the pass if any illegal. Alex operation could not be lowered.
            if (failed(mlir::applyPartialConversion(module, target, std::move(patterns))))
            {
                signalPassFailure();
            }
        }
    };
}

// Create and return the Alex-to-Arith lowering pass.
std::unique_ptr<mlir::Pass> createAlexToArithPass()
{
    return std::make_unique<AlexToArithPass>();
}

// Register the pass with MLIR.
void registerAlexToArithPass()
{
    mlir::PassRegistration<AlexToArithPass>();
}