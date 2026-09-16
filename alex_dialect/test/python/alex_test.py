import os
import sys
from pathlib import Path

BUILD_DIR = "/home/cflux/Documents/Jeoffrey/01.Projects/01.MLIR/MLIR/alex_dialect/build"

sys.path.insert(
    0,
    os.path.join(BUILD_DIR, "python_packages", "alex")
)

from mlir_alex.ir import Context, Module
from mlir_alex.passmanager import PassManager
from mlir_alex.dialects import alex_nanobind as alex


TEST_DIR = Path(__file__).parent.parent

test_files = [
    #"add_two_tensors.mlir",
    "expression.mlir",
    #"type_constraint.mlir",
]


with Context():
    alex.register_dialects()

    pm = PassManager.parse(
        "builtin.module(convert-alex-to-arith)"
    )

    for filename in test_files:
        path = TEST_DIR / filename

        print(f"\n========== {filename} ==========\n")

        mlir_text = path.read_text()

        module = Module.parse(mlir_text)

        print("--- Before Lowering ---")
        print(module)

        pm.run(module.operation)

        print("--- After Lowering ---")
        print(module)