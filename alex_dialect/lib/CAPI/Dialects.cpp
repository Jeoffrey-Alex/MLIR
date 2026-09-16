#include "Alex-c/Dialects.h"
#include "Alex/AlexDialect.h"

#include "mlir/CAPI/Registration.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Alex,alex,alex::AlexDialect)