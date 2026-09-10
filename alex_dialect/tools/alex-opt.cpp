#include "Alex/AlexDialect.h"
#include "Alex/AlexOps.h"
#include "mlir/Pass/Pass.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

std::unique_ptr<mlir::Pass> createAlexToArithPass();


int main(int argc, char **argv)
{

    // Register the custom lowering pass.
    mlir::registerPass([]() -> std::unique_ptr<mlir::Pass>
                       { return createAlexToArithPass(); });

    mlir::DialectRegistry registry;

    registry.insert<
        alex::AlexDialect,
        mlir::arith::ArithDialect,
        mlir::func::FuncDialect>();

    return mlir::asMainReturnCode(
        mlir::MlirOptMain(argc,argv,"Alex Optimizer\n", registry));
}