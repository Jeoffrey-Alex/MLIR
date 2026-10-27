// RUN: alex-opt %s -convert-alex-to-arith | FileCheck %s

// Only the op is converted. The function signature keeps !alex.bool, so the
// type converter joins both sides with unrealized_conversion_cast.
// CHECK-LABEL: func.func @aten_not
// CHECK-SAME: (%[[X:.*]]: !alex.bool) -> !alex.bool
// CHECK: %[[IN:.*]] = builtin.unrealized_conversion_cast %[[X]] : !alex.bool to i1
// CHECK: %[[TRUE:.*]] = arith.constant true
// CHECK: %[[NOT:.*]] = arith.xori %[[IN]], %[[TRUE]] : i1
// CHECK: %[[OUT:.*]] = builtin.unrealized_conversion_cast %[[NOT]] : i1 to !alex.bool
// CHECK: return %[[OUT]] : !alex.bool
// CHECK-NOT: alex.aten.__not__
func.func @aten_not(%x: !alex.bool) -> !alex.bool {
  %r = alex.aten.__not__ %x : !alex.bool -> !alex.bool
  return %r : !alex.bool
}
