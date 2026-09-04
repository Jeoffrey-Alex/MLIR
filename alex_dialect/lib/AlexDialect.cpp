#include "Alex/AlexDialect.h"
#include "Alex/AlexOps.h"

#include "mlir/IR/DialectImplementation.h"

#include "Alex/AlexDialect.cpp.inc"

void alex::AlexDialect::initialize() {
    addOperations<alex::AddOp>();
}