// RUN: alex-opt %s -convert-alex-to-arith | FileCheck %s

module {
  func.func @Addcmul()->tensor<2x2x2xf32> attributes{llvm.emit_c_interface} {
    %input = "alex.constant"(){
      value = dense<
          [ [ [ 1.0, 2.0 ], [ 3.0, 4.0 ] ], [ [ 5.0, 6.0 ], [ 7.0, 8.0 ] ] ]> :
          tensor<2x2x2xf32>
    } : ()
            ->tensor<2x2x2xf32>

              %tensor1 = "alex.constant"(){
      value = dense<
          [ [ [ 2.0, 3.0 ], [ 4.0, 5.0 ] ], [ [ 6.0, 7.0 ], [ 8.0, 9.0 ] ] ]> :
          tensor<2x2x2xf32>
    } : ()
            ->tensor<2x2x2xf32>

                          %tensor2 = "alex.constant"(){
      value = dense<[
        [ [ 5.0, 6.0 ], [ 7.0, 8.0 ] ], [ [ 9.0, 10.0 ], [ 11.0, 12.0 ] ]
      ]> : tensor<2x2x2xf32>
    } : ()
            ->tensor<2x2x2xf32>

                                      %result = "alex.addcmul"(
                                                   %input, %tensor1,
                                                   %tensor2){value = 2.0 : f32}
        : (tensor<2x2x2xf32>, tensor<2x2x2xf32>, tensor<2x2x2xf32>)
              ->tensor<2x2x2xf32>

          return %result : tensor<2x2x2xf32>
  }
}

// CHECK-LABEL: func.func @Addcmul
// CHECK: linalg.fill
// CHECK: linalg.elementwise kind=#linalg.elementwise_kind<mul>
// CHECK: linalg.elementwise kind=#linalg.elementwise_kind<mul>
// CHECK: linalg.elementwise kind=#linalg.elementwise_kind<add>
// CHECK-NOT: alex.addcmul