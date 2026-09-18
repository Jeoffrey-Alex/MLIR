#include "Alex-c/Dialects.h"
#include "Alex/AlexDialect.h"

#include "mlir/CAPI/Registration.h"

#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"

void alexRegisterAllExtensions(MlirDialectRegistry registry)
{
    mlir::DialectRegistry *reg = unwrap(registry);

    mlir::arith::registerBufferizableOpInterfaceExternalModels(*reg);

    mlir::bufferization::registerBufferizableOpInterfaceExternalModels(*reg);

    mlir::bufferization::func_ext::
        registerBufferizableOpInterfaceExternalModels(*reg);

    mlir::tensor::
        registerBufferizableOpInterfaceExternalModels(*reg);

    mlir::linalg::
        registerBufferizableOpInterfaceExternalModels(*reg);
}

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Alex, alex, alex::AlexDialect)