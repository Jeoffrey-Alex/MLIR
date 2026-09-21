#include "Alex/AlexDialect.h"
#include "Alex/AlexOps.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

std::unique_ptr<mlir::Pass> createAlexToArithPass();

int main(int argc, char **argv) {

  mlir::registerAllPasses();

  // Register the custom lowering pass.
  mlir::registerPass(
      []() -> std::unique_ptr<mlir::Pass> { return createAlexToArithPass(); });

  mlir::DialectRegistry registry;

  registry.insert<alex::AlexDialect, mlir::arith::ArithDialect,
                  mlir::func::FuncDialect, mlir::tensor::TensorDialect>();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Alex Optimizer\n", registry));
}