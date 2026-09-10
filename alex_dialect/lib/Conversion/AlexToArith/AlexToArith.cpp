#include "Alex/AlexOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
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
            // To handle Float Types
            if (llvm::isa<mlir::FloatType>(op.getInput1().getType()))
            {
                rewriter.replaceOpWithNewOp<mlir::arith::AddFOp>(op, adaptor.getInput1(), adaptor.getInput2());
            }

            // To handle Integer types
            if (llvm::isa<mlir::IntegerType>(op.getInput1().getType()))
            {
                rewriter.replaceOpWithNewOp<mlir::arith::AddIOp>(op, adaptor.getInput1(), adaptor.getInput2());
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

        //PASS name
        mlir::StringRef getArgument() const final
        {
            return "convert-alex-to-arith";
        }

        //Pass Description
        mlir::StringRef getDescription() const final
        {
            return "Lower Alex Operations to Arith operation";
        }

        // alex.add depends on this inbuilt dialect
        void getDependentDialects(mlir::DialectRegistry &registry) const override
        {
            registry.insert<mlir::arith::ArithDialect>();
        }

        // entry point of lowering pass
        void runOnOperation() override
        {
            mlir::MLIRContext &context = getContext();
            mlir::ModuleOp module = getOperation();

            //register leagal and illegal operations fro conversion
            mlir::ConversionTarget target(context);

            target.addLegalDialect<mlir::arith::ArithDialect>();
            target.addIllegalOp<alex::AddOp>();

            mlir::RewritePatternSet patterns(&context);

            patterns.add<ConvertAddOp, ConvertConstOp>(&context);

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