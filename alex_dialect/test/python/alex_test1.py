import os
import sys
import ctypes
from pathlib import Path

import torch

BUILD_DIR = (
    "/home/cflux/Documents/Jeoffrey/01.Projects/01.MLIR/"
    "MLIR/alex_dialect/build"
)

sys.path.insert(
    0,
    os.path.join(BUILD_DIR, "python_packages", "alex")
)

from mlir_alex.ir import Context, Module, RankedTensorType
from mlir_alex.passmanager import PassManager
from mlir_alex.execution_engine import ExecutionEngine
from mlir_alex.dialects import alex_nanobind as alex

TEST_DIR = Path(__file__).parent.parent
MLIR_FILE = TEST_DIR / "addcmul_2d_ds.mlir"

MLIR_TO_CTYPE = {
    "f32": ctypes.c_float,
    "f64": ctypes.c_double,
    "i8":  ctypes.c_int8,
    "i16": ctypes.c_int16,
    "i32": ctypes.c_int32,
    "i64": ctypes.c_int64,
}


def ctype_for(mlir_type):
    key = str(mlir_type)
    if key not in MLIR_TO_CTYPE:
        raise RuntimeError(f"Don't know the ctypes equivalent of '{key}'")
    return MLIR_TO_CTYPE[key]


def make_memref_struct(rank):
    if rank == 0:
        fields = [
            ("allocated", ctypes.c_void_p),
            ("aligned", ctypes.c_void_p),
            ("offset", ctypes.c_int64),
        ]
    else:
        fields = [
            ("allocated", ctypes.c_void_p),
            ("aligned", ctypes.c_void_p),
            ("offset", ctypes.c_int64),
            ("sizes", ctypes.c_int64 * rank),
            ("strides", ctypes.c_int64 * rank),
        ]
    return type("MemRef", (ctypes.Structure,), {"_fields_": fields})


def get_all_funcs(module):
    funcs = []
    for op in module.body.operations:
        if op.operation.name == "func.func":
            name = str(op.attributes["sym_name"]).strip('"')
            function_type = op.attributes["function_type"].value
            return_type = function_type.results[0]
            funcs.append((name, return_type))
    if not funcs:
        raise RuntimeError("No func.func found in this module")
    return funcs


def read_nested(data_ptr, sizes, strides, offset, dim=0, base_index=()):
    if dim == len(sizes):
        flat_index = offset + sum(i * s for i, s in zip(base_index, strides))
        return data_ptr[flat_index]
    return [
        read_nested(data_ptr, sizes, strides, offset, dim + 1, base_index + (i,))
        for i in range(sizes[dim])
    ]


def round_nested(value, digits=4):
    if isinstance(value, list):
        return [round_nested(v, digits) for v in value]
    if isinstance(value, float):
        return round(value, digits)
    return value


def run_tensor(module, func_name, rank, elem_ctype):
    ee = ExecutionEngine(module)

    MemRefN = make_memref_struct(rank)
    descriptor = MemRefN()
    descriptor_ptr = ctypes.pointer(descriptor)
    descriptor_ptr_ptr = ctypes.pointer(descriptor_ptr)

    ee.invoke(func_name, descriptor_ptr_ptr)

    data_ptr = ctypes.cast(descriptor.aligned, ctypes.POINTER(elem_ctype))

    if rank == 0:
        actual = data_ptr[descriptor.offset]
    else:
        sizes = list(descriptor.sizes)
        strides = list(descriptor.strides)
        actual = read_nested(data_ptr, sizes, strides, descriptor.offset)
        actual = round_nested(actual)

    if descriptor.allocated:
        libc = ctypes.CDLL("libc.so.6")
        libc.free.argtypes = [ctypes.c_void_p]
        libc.free.restype = None
        libc.free(descriptor.allocated)

    return actual


def test_addcmul():
    print("\n")
    print("=" * 60)
    print("TEST: addcmul")
    print("=" * 60)

    mlir_text = MLIR_FILE.read_text()

    input_val = [
        [1.0, 2.0]
    ]

    tensor1_val = [
        [2.0, 3.0]
    ]

    tensor2_val = [
        [5.0],
        [6.0]
    ]

    value = 2.0

    expected = torch.addcmul(
        torch.tensor(input_val, dtype=torch.float32),
        torch.tensor(tensor1_val, dtype=torch.float32),
        torch.tensor(tensor2_val, dtype=torch.float32),
        value=value,
    )

    print("\nExpected (PyTorch):")
    print(expected)

    with Context():
        alex.register_dialects()

        module = Module.parse(mlir_text)
        funcs = get_all_funcs(module)

        pm = PassManager.parse(
            "builtin.module("
            "convert-alex-to-arith,"
            "one-shot-bufferize{bufferize-function-boundaries=true},"
            "convert-linalg-to-loops,"
            "convert-scf-to-cf,"
            "convert-bufferization-to-memref,"
            "convert-arith-to-llvm,"
            "convert-index-to-llvm,"
            "finalize-memref-to-llvm,"
            "convert-func-to-llvm,"
            "convert-cf-to-llvm,"
            "reconcile-unrealized-casts"
            ")"
        )
        pm.run(module.operation)

        func_name, return_type = funcs[0]

        tensor_type = RankedTensorType(return_type)
        rank = len(tensor_type.shape)
        elem_ctype = ctype_for(tensor_type.element_type)

        actual = run_tensor(module, func_name, rank, elem_ctype)
        actual_tensor = torch.tensor(actual, dtype=torch.float32)

        print("\nActual (Alex):")
        print(actual_tensor)

        assert torch.allclose(actual_tensor, expected, rtol=1e-5, atol=1e-6), (
            "\nResult mismatch!\n"
            f"Expected:\n{expected}\n"
            f"Actual:\n{actual_tensor}"
        )

        print("\nPASS")