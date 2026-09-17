import os
import sys
import ctypes
from pathlib import Path

# Python binding setup

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


# Test files

TEST_DIR = Path(__file__).parent.parent

test_files = [
    #"add_scalar.mlir",
    "add_tensors_scalar.mlir",
    #"expression.mlir"
]

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

def run_scalar(module, func_name, elem_ctype):
    print("\n--- Execution Result ---")

    ee = ExecutionEngine(module)

    result = (elem_ctype * 1)(0)
    ee.invoke(func_name, result)

    print(f"{func_name}() =", result[0])

def run_tensor(module, func_name, rank, elem_ctype):
    print("\n--- Execution Result ---")

    ee = ExecutionEngine(module)

    MemRefN = make_memref_struct(rank)
    descriptor = MemRefN()
    descriptor_ptr = ctypes.pointer(descriptor)
    descriptor_ptr_ptr = ctypes.pointer(descriptor_ptr)

    ee.invoke(func_name, descriptor_ptr_ptr)

    data_ptr = ctypes.cast(descriptor.aligned, ctypes.POINTER(elem_ctype))

    if rank == 0:
        value = data_ptr[descriptor.offset]
        if isinstance(value, float):
            value = round(value, 4)
        print(f"\n{func_name}() =", value)
    else:
        sizes = list(descriptor.sizes)
        strides = list(descriptor.strides)
        nested = read_nested(data_ptr, sizes, strides, descriptor.offset)
        nested = round_nested(nested)
        print(f"\n{func_name}() =")
        print(nested)

    if descriptor.allocated:
        libc = ctypes.CDLL("libc.so.6")
        libc.free.argtypes = [ctypes.c_void_p]
        libc.free.restype = None
        libc.free(descriptor.allocated)

with Context():

    alex.register_dialects()

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

    for filename in test_files:

        path = TEST_DIR / filename

        print("\n==============================================")
        print(f"========== {filename} ==========")
        print("==============================================\n")

        mlir_text = path.read_text()
        module = Module.parse(mlir_text)

        print("--- Before Lowering ---")
        print(module)

        funcs = get_all_funcs(module)

        pm.run(module.operation)

        print("--- After Lowering ---")
        print(module)

        for func_name, return_type in funcs:

            try:
                tensor_type = RankedTensorType(return_type)
                rank = len(tensor_type.shape)
                elem_ctype = ctype_for(tensor_type.element_type)
            except Exception:
                rank = None
                elem_ctype = ctype_for(return_type)

            if rank is None:
                run_scalar(module, func_name, elem_ctype)
            else:
                run_tensor(module, func_name, rank, elem_ctype)