
#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

typedef unsigned char uchar;

void restore_image(uchar*, const float*, const float*, const uchar*,
        const float*, const int, const int, const int, const int);



#endif   /* End of CUDA_KERNELS.H */




