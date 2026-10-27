#include "Alex/AlexDialect.h"
#include "Alex/AlexOps.h"

#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include "Alex/AlexDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "Alex/AlexTypes.cpp.inc"

void alex::AlexDialect::initialize() {
  addOperations<alex::PractiseOp>();
  addOperations<alex::AddOp, alex::SubOp, alex::MulOp, alex::ConstOp,
                alex::reluOp, alex::transposeOp, alex::AddcmulOp,
                alex::addLinalgOp, alex::addBCLinalgOp, alex::reductionLinalgOp,
                alex::argminOp, alex::NotOp>();
  // clang-format off
  addTypes<
#define GET_TYPEDEF_LIST
#include "Alex/AlexTypes.cpp.inc"
      >();
  // clang-format on
}