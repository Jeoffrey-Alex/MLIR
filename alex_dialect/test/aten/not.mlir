// RUN: alex-opt %s | FileCheck %s

// The type and the op parse and print back unchanged.
// CHECK-LABEL: func.func @aten_not
// CHECK-SAME: (%[[X:.*]]: !alex.bool) -> !alex.bool
// CHECK: %[[R:.*]] = alex.aten.__not__ %[[X]] : !alex.bool -> !alex.bool
// CHECK: return %[[R]] : !alex.bool
func.func @aten_not(%x: !alex.bool) -> !alex.bool {
  %r = alex.aten.__not__ %x : !alex.bool -> !alex.bool
  return %r : !alex.bool
}
