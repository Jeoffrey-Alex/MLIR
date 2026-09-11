module {

    func.func @Scalar_int() -> i32 {
        %c = "alex.constant"(){value = 3 : i32}:() -> i32
        %d = "alex.constant"(){value = 4 : i32}:() -> i32

        %result_int = "alex.add"(%c,%d) : (i32,i32) -> i32

        return %result_int : i32
    }

    func.func @Scalar_float() -> f32 {
        %a = "alex.constant"(){value = 1.0 : f32}:() -> f32
        %b = "alex.constant"(){value = 2.0 : f32}:() -> f32


        %result_float = "alex.add"(%a,%b) : (f32,f32) -> f32


        return %result_float : f32
    }
}