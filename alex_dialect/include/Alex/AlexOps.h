#ifndef ALEX_OPS_H
#define ALEX_OPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"     


#include "Alex/AlexDialect.h"

#define GET_OP_CLASSES
#include "Alex/AlexOps.h.inc"

namespace alex {

/// Turns a torch-style dim (negative counts from the end) into 0..rank-1.
/// Fails if the dim is out of range.
mlir::FailureOr<int64_t> normalizeDim(int64_t dim, int64_t rank);

/// Result type of swapping dim0 and dim1 of inputType.
/// Fails if either dim is out of range.
mlir::FailureOr<mlir::RankedTensorType>
inferTransposeType(mlir::RankedTensorType inputType, int64_t dim0,
                   int64_t dim1);

} // namespace alex

#endif
