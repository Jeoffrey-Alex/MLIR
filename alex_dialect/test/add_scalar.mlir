module {
    func.func @Scalar() -> f32 attributes { llvm.emit_c_interface } {
        %a = "alex.constant"(){value = 1 : i32}:() -> i32
        %b = "alex.constant"(){value = 2.0 : f32}:() -> f32
        %c = "alex.constant"(){value = 3.0 : f32}:() -> f32
        %d = "alex.constant"(){value = 4.0 : f32}:() -> f32

        %bc = "alex.mul"(%b,%c) : (f32, f32) -> f32
        %abc = "alex.add"(%a,%bc) : (i32,f32) -> f32
        %result = "alex.sub"(%abc,%d) : (f32, f32) -> f32

        return %result : f32
    }
}