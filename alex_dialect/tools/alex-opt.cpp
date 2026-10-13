#include "Alex/AlexDialect.h"
#include "Alex/AlexOps.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"

std::unique_ptr<mlir::Pass> createAlexToArithPass();
std::unique_ptr<mlir::Pass> createAlexToLinalgPass();

int main(int argc, char **argv)
{

    mlir::registerAllPasses();

    // Register the custom lowering pass.
    mlir::registerPass(
        []() -> std::unique_ptr<mlir::Pass>
        { return createAlexToArithPass(); });
    mlir::registerPass(
        []() -> std::unique_ptr<mlir::Pass>
        { return createAlexToLinalgPass(); });

    mlir::DialectRegistry registry;

    registry.insert<alex::AlexDialect, mlir::arith::ArithDialect,
                    mlir::func::FuncDialect, mlir::tensor::TensorDialect>();

    mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::bufferization::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
        registry);
    mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);

    return mlir::asMainReturnCode(
        mlir::MlirOptMain(argc, argv, "Alex Optimizer\n", registry));
}