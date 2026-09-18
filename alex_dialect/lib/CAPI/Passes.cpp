#include "Alex-c/Passes.h"
#include "Alex/AlexDialect.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"

void registerAlexToArithPass();

void alexRegisterAllPasses()
{
    registerAlexToArithPass();

    mlir::bufferization::registerOneShotBufferizePass();
    mlir::registerConvertLinalgToLoopsPass();
    mlir::registerSCFToControlFlowPass();
    mlir::registerConvertBufferizationToMemRefPass();

    mlir::registerArithToLLVMConversionPass();
    mlir::registerConvertIndexToLLVMPass();
    mlir::registerFinalizeMemRefToLLVMConversionPass();
    mlir::registerConvertControlFlowToLLVMPass();

    mlir::registerConvertFuncToLLVMPass();
    mlir::registerReconcileUnrealizedCastsPass();
}