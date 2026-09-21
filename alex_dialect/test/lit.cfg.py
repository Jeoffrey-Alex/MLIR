import os
import lit.formats

config.name = "Alex"
config.test_format = lit.formats.ShTest()
config.suffixes = [".mlir"]

config.environment["PATH"] = (
    "/home/cflux/Documents/Jeoffrey/01.Projects/01.MLIR/MLIR/alex_dialect/build/tools"
    + os.pathsep
    + config.environment.get("PATH", "")
)