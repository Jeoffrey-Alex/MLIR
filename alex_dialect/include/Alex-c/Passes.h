#ifndef ALEX_C_PASSES_H
#define ALEX_C_PASSES_H

#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_CAPI_EXPORTED void alexRegisterAllPasses();

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // ALEX_C_PASSES_H
