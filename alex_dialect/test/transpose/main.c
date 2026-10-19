#include <stdint.h>
#include <stdio.h>

typedef struct {
    void *allocated;
    void *aligned;
    int64_t offset;
    int64_t sizes[2];
    int64_t strides[2];
} MemRef2D;

extern MemRef2D transpose_f32(void);

int main(void)
{
    MemRef2D result = transpose_f32();
    float *data = (float *)result.aligned;

    for (int i = 0; i < result.sizes[0]; i++) {
        for (int j = 0; j < result.sizes[1]; j++) {
            printf("%g ", data[result.offset +
                               i * result.strides[0] +
                               j * result.strides[1]]);
        }
        printf("\n");
    }

    return 0;
}