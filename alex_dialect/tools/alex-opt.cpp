#include "Alex/AlexDialect.h"
#include "Alex/AlexOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv)
{
    mlir::DialectRegistry registry;

    registry.insert<
        alex::AlexDialect,
        mlir::arith::ArithDialect,
        mlir::func::FuncDialect>();

    return mlir::asMainReturnCode(
        mlir::MlirOptMain(argc,argv,"Alex Optimizer\n", registry)
    );
}