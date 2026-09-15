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
    class ConvertAddOp : public mlir::OpConversionPattern<alex::AddOp>
    {
    public:
        using mlir::OpConversionPattern<alex::AddOp>::OpConversionPattern;

        mlir::LogicalResult matchAndRewrite(alex::AddOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter) const override
        {
            // float + float
            if ((llvm::isa<mlir::FloatType>(op.getInput1().getType())) && (llvm::isa<mlir::FloatType>(op.getInput2().getType())))
            {
                rewriter.replaceOpWithNewOp<mlir::arith::AddFOp>(op, adaptor.getInput1(), adaptor.getInput2());
            }
            // int + int
            else if ((llvm::isa<mlir::IntegerType>(op.getInput1().getType())) && (llvm::isa<mlir::IntegerType>(op.getInput2().getType())))
            {
                rewriter.replaceOpWithNewOp<mlir::arith::AddIOp>(op, adaptor.getInput1(), adaptor.getInput2());
            }
            // tensor + tensor
            else if ((llvm::isa<mlir::RankedTensorType>(op.getInput1().getType())) && (llvm::isa<mlir::RankedTensorType>(op.getInput2().getType())))
            {
                auto input1Type = llvm::cast<mlir::RankedTensorType>(op.getInput1().getType());
                auto input2Type = llvm::cast<mlir::RankedTensorType>(op.getInput2().getType());

                // Both tensors must have the same shape
                if (input1Type != input2Type)
                    return mlir::failure();

                auto resultType = llvm::cast<mlir::RankedTensorType>(op.getResult().getType());

                auto emptyTensor = mlir::tensor::EmptyOp::create(rewriter, op.getLoc(), resultType.getShape(), resultType.getElementType());

                auto kindAttr = mlir::linalg::ElementwiseKindAttr::get(rewriter.getContext(), mlir::linalg::ElementwiseKind::add);

                auto indexingMaps = rewriter.getAffineMapArrayAttr({mlir::AffineMap::getMultiDimIdentityMap(
                                                                        resultType.getRank(), rewriter.getContext()),
                                                                    mlir::AffineMap::getMultiDimIdentityMap(
                                                                        resultType.getRank(), rewriter.getContext()),
                                                                    mlir::AffineMap::getMultiDimIdentityMap(
                                                                        resultType.getRank(), rewriter.getContext())});

                auto elementwiseOp = mlir::linalg::ElementwiseOp::create(rewriter, op.getLoc(), mlir::ValueRange{adaptor.getInput1(), adaptor.getInput2()}, mlir::ValueRange{emptyTensor}, kindAttr, indexingMaps);

                rewriter.replaceOp(op, elementwiseOp.getResults());

                return mlir::success();
            }
            // Unsupported combinations:
            // int + float, float + int,
            // int + tensor, tensor + int,
            // float + tensor, tensor + float,
            // tensors with different shapes
            else
            {
                return mlir::failure();
            }

            return mlir::success();
        }
    };

    class ConvertSubOp : public mlir::OpConversionPattern<alex::SubOp>
    {
    public:
        using mlir::OpConversionPattern<alex::SubOp>::OpConversionPattern;

        mlir::LogicalResult matchAndRewrite(alex::SubOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter) const override
        {
            if ((llvm::isa<mlir::FloatType>(op.getInput1().getType())) && (llvm::isa<mlir::FloatType>(op.getInput2().getType())))
            {
                rewriter.replaceOpWithNewOp<mlir::arith::SubFOp>(op, adaptor.getInput1(), adaptor.getInput2());
            }
            else if ((llvm::isa<mlir::IntegerType>(op.getInput1().getType())) && (llvm::isa<mlir::IntegerType>(op.getInput2().getType())))
            {
                rewriter.replaceOpWithNewOp<mlir::arith::SubIOp>(op, adaptor.getInput1(), adaptor.getInput2());
            }
            else if ((llvm::isa<mlir::RankedTensorType>(op.getInput1().getType())) && (llvm::isa<mlir::RankedTensorType>(op.getInput2().getType())))
            {
                auto input1Type = llvm::cast<mlir::RankedTensorType>(op.getInput1().getType());
                auto input2Type = llvm::cast<mlir::RankedTensorType>(op.getInput2().getType());

                // Both tensors must have the same shape
                if (input1Type != input2Type)
                    return mlir::failure();

                auto resultType = llvm::cast<mlir::RankedTensorType>(op.getResult().getType());

                auto emptyTensor = mlir::tensor::EmptyOp::create(rewriter, op.getLoc(), resultType.getShape(), resultType.getElementType());

                auto kindAttr = mlir::linalg::ElementwiseKindAttr::get(rewriter.getContext(), mlir::linalg::ElementwiseKind::sub);

                auto indexingMaps = rewriter.getAffineMapArrayAttr({mlir::AffineMap::getMultiDimIdentityMap(
                                                                        resultType.getRank(), rewriter.getContext()),
                                                                    mlir::AffineMap::getMultiDimIdentityMap(
                                                                        resultType.getRank(), rewriter.getContext()),
                                                                    mlir::AffineMap::getMultiDimIdentityMap(
                                                                        resultType.getRank(), rewriter.getContext())});

                auto elementwiseOp = mlir::linalg::ElementwiseOp::create(rewriter, op.getLoc(), mlir::ValueRange{adaptor.getInput1(), adaptor.getInput2()}, mlir::ValueRange{emptyTensor}, kindAttr, indexingMaps);

                rewriter.replaceOp(op, elementwiseOp.getResults());

                return mlir::success();
            }
            else
            {
                return mlir::failure();
            }

            return mlir::success();
        }
    };

    class ConvertMulOp : public mlir::OpConversionPattern<alex::MulOp>
    {
    public:
        using mlir::OpConversionPattern<alex::MulOp>::OpConversionPattern;

        mlir::LogicalResult matchAndRewrite(alex::MulOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter) const override
        {
            // To handle Float Types
            if ((llvm::isa<mlir::FloatType>(op.getInput1().getType())) && (llvm::isa<mlir::FloatType>(op.getInput2().getType())))
            {
                rewriter.replaceOpWithNewOp<mlir::arith::MulFOp>(op, adaptor.getInput1(), adaptor.getInput2());
            }
            else if ((llvm::isa<mlir::IntegerType>(op.getInput1().getType())) && (llvm::isa<mlir::IntegerType>(op.getInput2().getType())))
            {
                rewriter.replaceOpWithNewOp<mlir::arith::MulIOp>(op, adaptor.getInput1(), adaptor.getInput2());
            }
            else if ((llvm::isa<mlir::RankedTensorType>(op.getInput1().getType())) && (llvm::isa<mlir::RankedTensorType>(op.getInput2().getType())))
            {
                auto input1Type = llvm::cast<mlir::RankedTensorType>(op.getInput1().getType());
                auto input2Type = llvm::cast<mlir::RankedTensorType>(op.getInput2().getType());

                // Both tensors must have the same shape
                if (input1Type != input2Type)
                    return mlir::failure();

                auto resultType = llvm::cast<mlir::RankedTensorType>(op.getResult().getType());

                auto emptyTensor = mlir::tensor::EmptyOp::create(rewriter, op.getLoc(), resultType.getShape(), resultType.getElementType());

                auto kindAttr = mlir::linalg::ElementwiseKindAttr::get(rewriter.getContext(), mlir::linalg::ElementwiseKind::mul);

                auto indexingMaps = rewriter.getAffineMapArrayAttr({mlir::AffineMap::getMultiDimIdentityMap(
                                                                        resultType.getRank(), rewriter.getContext()),
                                                                    mlir::AffineMap::getMultiDimIdentityMap(
                                                                        resultType.getRank(), rewriter.getContext()),
                                                                    mlir::AffineMap::getMultiDimIdentityMap(
                                                                        resultType.getRank(), rewriter.getContext())});

                auto elementwiseOp = mlir::linalg::ElementwiseOp::create(rewriter, op.getLoc(), mlir::ValueRange{adaptor.getInput1(), adaptor.getInput2()}, mlir::ValueRange{emptyTensor}, kindAttr, indexingMaps);

                rewriter.replaceOp(op, elementwiseOp.getResults());

                return mlir::success();
            }
            else
            {
                return mlir::failure();
            }

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

            mlir::RewritePatternSet patterns(&context);

            patterns.add<ConvertAddOp, ConvertConstOp, ConvertSubOp, ConvertMulOp>(&context);

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