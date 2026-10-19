; ModuleID = '../test/transpose/main.c'
source_filename = "../test/transpose/main.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.MemRef2D = type { ptr, ptr, i64, [2 x i64], [2 x i64] }

@.str = private unnamed_addr constant [4 x i8] c"%g \00", align 1
@.str.1 = private unnamed_addr constant [2 x i8] c"\0A\00", align 1

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @main() #0 {
  %1 = alloca i32, align 4
  %2 = alloca %struct.MemRef2D, align 8
  %3 = alloca ptr, align 8
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  store i32 0, ptr %1, align 4
  call void @transpose_f32(ptr dead_on_unwind writable sret(%struct.MemRef2D) align 8 %2)
  %6 = getelementptr inbounds nuw %struct.MemRef2D, ptr %2, i32 0, i32 1
  %7 = load ptr, ptr %6, align 8
  store ptr %7, ptr %3, align 8
  store i32 0, ptr %4, align 4
  br label %8

8:                                                ; preds = %50, %0
  %9 = load i32, ptr %4, align 4
  %10 = sext i32 %9 to i64
  %11 = getelementptr inbounds nuw %struct.MemRef2D, ptr %2, i32 0, i32 3
  %12 = getelementptr inbounds [2 x i64], ptr %11, i64 0, i64 0
  %13 = load i64, ptr %12, align 8
  %14 = icmp slt i64 %10, %13
  br i1 %14, label %15, label %53

15:                                               ; preds = %8
  store i32 0, ptr %5, align 4
  br label %16

16:                                               ; preds = %45, %15
  %17 = load i32, ptr %5, align 4
  %18 = sext i32 %17 to i64
  %19 = getelementptr inbounds nuw %struct.MemRef2D, ptr %2, i32 0, i32 3
  %20 = getelementptr inbounds [2 x i64], ptr %19, i64 0, i64 1
  %21 = load i64, ptr %20, align 8
  %22 = icmp slt i64 %18, %21
  br i1 %22, label %23, label %48

23:                                               ; preds = %16
  %24 = load ptr, ptr %3, align 8
  %25 = getelementptr inbounds nuw %struct.MemRef2D, ptr %2, i32 0, i32 2
  %26 = load i64, ptr %25, align 8
  %27 = load i32, ptr %4, align 4
  %28 = sext i32 %27 to i64
  %29 = getelementptr inbounds nuw %struct.MemRef2D, ptr %2, i32 0, i32 4
  %30 = getelementptr inbounds [2 x i64], ptr %29, i64 0, i64 0
  %31 = load i64, ptr %30, align 8
  %32 = mul nsw i64 %28, %31
  %33 = add nsw i64 %26, %32
  %34 = load i32, ptr %5, align 4
  %35 = sext i32 %34 to i64
  %36 = getelementptr inbounds nuw %struct.MemRef2D, ptr %2, i32 0, i32 4
  %37 = getelementptr inbounds [2 x i64], ptr %36, i64 0, i64 1
  %38 = load i64, ptr %37, align 8
  %39 = mul nsw i64 %35, %38
  %40 = add nsw i64 %33, %39
  %41 = getelementptr inbounds float, ptr %24, i64 %40
  %42 = load float, ptr %41, align 4
  %43 = fpext float %42 to double
  %44 = call i32 (ptr, ...) @printf(ptr noundef @.str, double noundef %43)
  br label %45

45:                                               ; preds = %23
  %46 = load i32, ptr %5, align 4
  %47 = add nsw i32 %46, 1
  store i32 %47, ptr %5, align 4
  br label %16, !llvm.loop !5

48:                                               ; preds = %16
  %49 = call i32 (ptr, ...) @printf(ptr noundef @.str.1)
  br label %50

50:                                               ; preds = %48
  %51 = load i32, ptr %4, align 4
  %52 = add nsw i32 %51, 1
  store i32 %52, ptr %4, align 4
  br label %8, !llvm.loop !7

53:                                               ; preds = %8
  ret i32 0
}

declare void @transpose_f32(ptr dead_on_unwind writable sret(%struct.MemRef2D) align 8) #1

declare i32 @printf(ptr noundef, ...) #1

attributes #0 = { noinline nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }

!llvm.module.flags = !{!0, !1, !2, !3}
!llvm.ident = !{!4}

!0 = !{i32 8, !"PIC Level", i32 2}
!1 = !{i32 7, !"PIE Level", i32 2}
!2 = !{i32 7, !"uwtable", i32 2}
!3 = !{i32 7, !"frame-pointer", i32 2}
!4 = !{!"clang version 23.0.0git (https://github.com/llvm/llvm-project.git dbe5e3391c0f7ecb4dd5b3f8943c1ebf689e4118)"}
!5 = distinct !{!5, !6}
!6 = !{!"llvm.loop.mustprogress"}
!7 = distinct !{!7, !6}
