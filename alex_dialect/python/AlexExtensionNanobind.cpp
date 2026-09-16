#include "Alex-c/Dialects.h"
#include "Alex-c/Passes.h"

#include "mlir/InitAllPasses.h"
#include "mlir-c/Dialect/Arith.h"
#include "mlir-c/Dialect/Bufferization.h"
#include "mlir-c/Dialect/ControlFlow.h"
#include "mlir-c/Dialect/Func.h"
#include "mlir-c/Dialect/LLVM.h"
#include "mlir-c/Dialect/Linalg.h"
#include "mlir-c/Dialect/Math.h"
#include "mlir-c/Dialect/MemRef.h"
#include "mlir-c/Dialect/SCF.h"
#include "mlir-c/Dialect/Tensor.h"

#include "mlir/Bindings/Python/IRCore.h"
#include "mlir/Bindings/Python/Nanobind.h"


namespace nb = nanobind;

NB_MODULE(_alexDialectsNanobind, m) {
  auto alexM = m.def_submodule("alex");

  mlirRegisterAllPasses();

  alexM.def(
      "register_dialects",
      [](mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::DefaultingPyMlirContext
             context,
         bool load) {
        MlirContext ctx = context.get()->get();

        MlirDialectRegistry registry = mlirDialectRegistryCreate();

        MlirDialectHandle handles[] = {
            mlirGetDialectHandle__alex__(),
            mlirGetDialectHandle__arith__(),
            mlirGetDialectHandle__func__(),
            mlirGetDialectHandle__math__(),
            mlirGetDialectHandle__memref__(),
            mlirGetDialectHandle__scf__(),
            mlirGetDialectHandle__cf__(),
            mlirGetDialectHandle__linalg__(),
            mlirGetDialectHandle__bufferization__(),
            mlirGetDialectHandle__tensor__(),
            mlirGetDialectHandle__llvm__(),
        };

        for (MlirDialectHandle handle : handles)
          mlirDialectHandleInsertDialect(handle, registry);

        mlirContextAppendDialectRegistry(ctx, registry);
        mlirDialectRegistryDestroy(registry);

        if (load) {
          for (MlirDialectHandle handle : handles)
            mlirDialectHandleLoadDialect(handle, ctx);
        }
      },
      nb::arg("context").none() = nb::none(),
      nb::arg("load") = true);
}