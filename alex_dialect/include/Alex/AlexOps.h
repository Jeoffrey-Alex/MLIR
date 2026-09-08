#ifndef ALEX_OPS_H
#define ALEX_OPS_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"

#include "Alex/AlexDialect.h"

#define GET_OP_CLASSES
#include "Alex/AlexOps.h.inc"

#endif
