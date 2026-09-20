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

        bool isFloat(mlir::Type type) const
        {
            return llvm::isa<mlir::FloatType>(type);
        }

        bool isInt(mlir::Type type) const
        {
            return llvm::isa<mlir::IntegerType>(type);
        }

        bool isTensor(mlir::Type type) const
        {
            return llvm::isa<mlir::RankedTensorType>(type);
        }

        // float op float
        mlir::LogicalResult lowerFloat(AlexOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter) const
        {
            rewriter.replaceOpWithNewOp<FloatOp>(op, adaptor.getInput1(), adaptor.getInput2());

            return mlir::success();
        }

        // int op int
        mlir::LogicalResult lowerInt(AlexOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter) const
        {
            rewriter.replaceOpWithNewOp<IntOp>(op, adaptor.getInput1(), adaptor.getInput2());

            return mlir::success();
        }

        // int op float
        mlir::LogicalResult lowerIntFloat(AlexOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter) const
        {
            auto floatType = llvm::cast<mlir::FloatType>(op.getInput2().getType());

            auto convertedInt = mlir::arith::SIToFPOp::create(rewriter, op.getLoc(), floatType, adaptor.getInput1());

            rewriter.replaceOpWithNewOp<FloatOp>(op, convertedInt, adaptor.getInput2());

            return mlir::success();
        }

        // float op int
        mlir::LogicalResult lowerFloatInt(AlexOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter) const
        {
            auto floatType = llvm::cast<mlir::FloatType>(op.getInput1().getType());

            auto convertedInt = mlir::arith::SIToFPOp::create(rewriter, op.getLoc(), floatType, adaptor.getInput2());

            rewriter.replaceOpWithNewOp<FloatOp>(op, adaptor.getInput1(), convertedInt);

            return mlir::success();
        }

        mlir::LogicalResult lowerScalarTensor(AlexOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter) const
        {
            auto input1Type = op.getInput1().getType();
            auto input2Type = op.getInput2().getType();

            mlir::Value tensor;
            mlir::Value scalar;
            mlir::RankedTensorType tensorType;

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

            auto emptyTensor = mlir::tensor::EmptyOp::create(rewriter, op.getLoc(), tensorType.getShape(), tensorType.getElementType());

            auto fillOp = mlir::linalg::FillOp::create(rewriter, op.getLoc(), scalar, emptyTensor.getResult());

            if (isTensor(input1Type))
            {
                return lowerTensor(op, tensor, fillOp.getResult(0), tensorType, rewriter);
            }

            return lowerTensor(op, fillOp.getResult(0), tensor, tensorType, rewriter);
        }

        mlir::LogicalResult lowerTensor(AlexOp op, mlir::Value input1, mlir::Value input2, mlir::RankedTensorType tensorType, mlir::ConversionPatternRewriter &rewriter) const
        {
            auto resultTensor = mlir::tensor::EmptyOp::create(rewriter, op.getLoc(), tensorType.getShape(), tensorType.getElementType());
            auto kindAttr = mlir::linalg::ElementwiseKindAttr::get(rewriter.getContext(), ElementwiseKind);
            auto identityMap = mlir::AffineMap::getMultiDimIdentityMap(tensorType.getRank(), rewriter.getContext());
            auto indexingMaps = rewriter.getAffineMapArrayAttr({identityMap, identityMap, identityMap});
            auto elementwiseOp = mlir::linalg::ElementwiseOp::create(rewriter, op.getLoc(), mlir::ValueRange{input1, input2}, mlir::ValueRange{resultTensor.getResult()}, kindAttr, indexingMaps);
            rewriter.replaceOp(op, elementwiseOp.getResults());
            return mlir::success();
        }

        // tensor + tensor
        mlir::LogicalResult lowerTensorTensor(AlexOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter) const
        {
            auto input1Type = llvm::cast<mlir::RankedTensorType>(op.getInput1().getType());
            auto input2Type = llvm::cast<mlir::RankedTensorType>(op.getInput2().getType());
            if (input1Type != input2Type)
                return mlir::failure();
            return lowerTensor(op, adaptor.getInput1(), adaptor.getInput2(), input1Type, rewriter);
        }

        // Main rewrite function
        mlir::LogicalResult matchAndRewrite(AlexOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter) const override
        {
            auto input1Type = op.getInput1().getType();

            auto input2Type = op.getInput2().getType();

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

    using ConvertAddOp = ConvertBinaryOp<alex::AddOp, mlir::arith::AddFOp, mlir::arith::AddIOp, mlir::linalg::ElementwiseKind::add>;

    using ConvertSubOp = ConvertBinaryOp<alex::SubOp, mlir::arith::SubFOp, mlir::arith::SubIOp, mlir::linalg::ElementwiseKind::sub>;

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
            auto value = llvm::dyn_cast<mlir::TypedAttr>(op.getValueAttr());

            if (!value)
                return mlir::failure();

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

        // alex.add depends on this inbuilt dialect
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

            target.addLegalDialect<mlir::arith::ArithDialect, mlir::linalg::LinalgDialect, mlir::tensor::TensorDialect>();
            target.addIllegalOp<alex::AddOp>();
            target.addIllegalOp<alex::AddcmulOp>();

            mlir::RewritePatternSet patterns(&context);

            patterns.add<ConvertAddOp, ConvertConstOp, ConvertSubOp, ConvertMulOp, ConvertAddcmulOp>(&context);

            if (failed(mlir::applyPartialConversion(module, target, std::move(patterns))))
            {
                signalPassFailure();
            }
        }
    };
}

std::unique_ptr<mlir::Pass> createAlexToArithPass()
{
    return std::make_unique<AlexToArithPass>();
}

void registerAlexToArithPass()
{
    mlir::PassRegistration<AlexToArithPass>();
}