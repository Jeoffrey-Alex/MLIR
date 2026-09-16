#ifndef ALEX_C_DIALECTS_H
#define ALEX_C_DIALECTS_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C"
{
#endif

    MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Alex, alex);

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // ALEX_C_DIALECTS_H
