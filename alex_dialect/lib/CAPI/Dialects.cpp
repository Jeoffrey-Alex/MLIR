#include "Alex-c/Dialects.h"
#include "Alex/AlexDialect.h"

#include "mlir/CAPI/Registration.h"

#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/BufferizableOpInterfaceImpl.h"

void alexRegisterAllExtensions(MlirDialectRegistry registry)
{
    mlir::DialectRegistry *reg = unwrap(registry);

    mlir::arith::registerBufferizableOpInterfaceExternalModels(*reg);
    mlir::bufferization::registerBufferizableOpInterfaceExternalModels(*reg);
}

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Alex, alex, alex::AlexDialect)