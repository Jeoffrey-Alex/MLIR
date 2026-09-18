module {
    func.func @AddScalar() -> f32 attributes { llvm.emit_c_interface } {
        %a = "alex.constant"() {value = {{INPUT1}} : f32} : () -> f32

        %b = "alex.constant"() {value = {{INPUT2}} : f32} : () -> f32

        %result = "alex.add"(%a, %b) :(f32, f32) -> f32

        return %result : f32
    }
}