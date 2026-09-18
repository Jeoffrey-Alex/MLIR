import os
import sys
import ctypes
from pathlib import Path

import pytest
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

    return type("MemRef",(ctypes.Structure,),{"_fields_": fields})

def get_all_funcs(module):
    funcs = []

    for op in module.body.operations:
        if op.operation.name == "func.func":
            name = str(op.attributes["sym_name"]).strip('"')

            function_type = (op.attributes["function_type"].value)

            return_type = function_type.results[0]

            funcs.append((name, return_type))

    if not funcs:
        raise RuntimeError("No func.func found in this module")

    return funcs

def read_nested(data_ptr,sizes,strides,offset,dim=0,base_index=()):
    if dim == len(sizes):
        flat_index = (offset+ sum(i * s
                for i, s in zip(base_index, strides)))

        return data_ptr[flat_index]

    return [
        read_nested(data_ptr,sizes,strides,offset,dim + 1,base_index + (i,))
        for i in range(sizes[dim])
    ]


def round_nested(value, digits=4):
    if isinstance(value, list):
        return [
            round_nested(v, digits)
            for v in value
        ]

    if isinstance(value, float):
        return round(value, digits)

    return value

def run_scalar(module, func_name, elem_ctype):
    ee = ExecutionEngine(module)

    result = (elem_ctype * 1)(0)

    ee.invoke(func_name,result)

    return result[0]

def run_tensor(module,func_name,rank,elem_ctype):
    ee = ExecutionEngine(module)

    MemRefN = make_memref_struct(rank)

    descriptor = MemRefN()

    descriptor_ptr = ctypes.pointer(descriptor)

    descriptor_ptr_ptr = ctypes.pointer(descriptor_ptr)

    # Execute lowered MLIR.
    ee.invoke(func_name,descriptor_ptr_ptr)

    data_ptr = ctypes.cast(descriptor.aligned,ctypes.POINTER(elem_ctype))

    if rank == 0:

        actual = data_ptr[descriptor.offset]

    else:

        sizes = list(descriptor.sizes)
        
        strides = list(descriptor.strides)

        actual = read_nested(data_ptr,sizes,strides,descriptor.offset)

        actual = round_nested(actual)

    # Free MLIR allocated memory.
    if descriptor.allocated:

        libc = ctypes.CDLL("libc.so.6")

        libc.free.argtypes = [ctypes.c_void_p]

        libc.free.restype = None

        libc.free(descriptor.allocated)

    return actual


def format_mlir_tensor(value):

    if isinstance(value, list):
        return "[" + ", ".join(format_mlir_tensor(v)
            for v in value) + "]"

    return str(float(value))


def pytorch_add(input1, input2):

    input1_tensor = torch.tensor(input1,dtype=torch.float32)

    input2_tensor = torch.tensor(input2,dtype=torch.float32)

    return torch.add(input1_tensor,input2_tensor)


@pytest.mark.parametrize(
    "op_type, input1, input2",
    [
        (
            "tensor_tensor",
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ),


        (
            "float_float",
            10.5,
            20.5,
        ),
        
        (
            "tensor_scalar",
            [[1.0, 2.0], [3.0, 4.0]],
            10.5,
        )

    ],
)
def test_add(op_type, input1, input2):

    print("\n")
    print("=" * 60)
    print(f"TEST: {op_type}")
    print("=" * 60)

    template_files = {
        "tensor_tensor": "add_tensors.mlir",
        "float_float": "add_scalar.mlir",
        "tensor_scalar": "add_tensors1.mlir",
    }

    if op_type not in template_files:
        raise RuntimeError(f"Unknown operation type: {op_type}")

    path = TEST_DIR / template_files[op_type]
    template = path.read_text()

    if op_type in ("tensor_tensor", "float_float", "tensor_scalar"):
        mlir_text = template.replace( "{{INPUT1}}",format_mlir_tensor(input1) if isinstance(input1, list) else str(input1),)
        mlir_text = mlir_text.replace("{{INPUT2}}",format_mlir_tensor(input2) if isinstance(input2, list) else str(input2),)

    expected = pytorch_add(input1, input2)

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

        for func_name, return_type in funcs:

            return_type_str = str(return_type)

            if return_type_str.startswith("tensor<"):
                tensor_type = RankedTensorType(return_type)

                rank = len(tensor_type.shape)
                
                elem_ctype = ctype_for(tensor_type.element_type)

                actual = run_tensor(module,func_name,rank,elem_ctype)

                actual_tensor = torch.tensor(actual,dtype=torch.float32)

            else:
                elem_ctype = ctype_for(return_type)

                actual = run_scalar(module,func_name,elem_ctype)

                actual_tensor = torch.tensor(actual,dtype=expected.dtype)

            print("\nActual (Alex):")
            print(actual_tensor)


            assert torch.allclose(actual_tensor,expected,rtol=1e-5,atol=1e-6)
            (
                "\nResult mismatch!\n"
                f"Expected:\n{expected}\n"
                f"Actual:\n{actual_tensor}"
            )

            print("\nPASS: Alex result matches PyTorch")
