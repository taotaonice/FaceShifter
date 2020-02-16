#include "cuda_kernels.hpp"

__global__ void restore_kernel(uchar* ret, const float* Yst, const float* mask,
        const uchar* Xt, const float* trans,
        const int H, const int W, const int h, const int w) {
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    int offset = blockDim.x * gridDim.x;
    while (tid < h * w) {
        int x = tid % w;
        int y = tid / w;
        float sx = x*trans[0] + y*trans[1] + trans[2];
        float sy = x*trans[3] + y*trans[4] + trans[5];
        if (sx < 0 || sy < 0 || sx >= W-1 || sy >= H-1){
            ret[tid*3+0] = Xt[tid*3+0];
            ret[tid*3+1] = Xt[tid*3+1];
            ret[tid*3+2] = Xt[tid*3+2];
            tid += offset;
            continue;
        }

        float xp = sx - (int)sx;
        float yp = sy - (int)sy;
        float color[3] = {0};
        for(int i=0;i<3;i++){
            float v = 0;
            float a = Yst[i*H*W + int(sy)*W + (int)(sx)];
            float b = Yst[i*H*W + int(sy)*W + (int)(sx+1)];
            float c = Yst[i*H*W + int(sy+1)*W + (int)(sx)];
            float d = Yst[i*H*W + int(sy+1)*W + (int)(sx+1)];
            float x1 = a + (b-a)*xp; float x2 = c + (d-c)*xp;
            v = x1 + (x2-x1)*yp;
            color[i] = v * 0.5 + 0.5;
        }
        float alpha = 0;
        {
            float a = mask[(int)(sy)*W + (int)(sx)];
            float b = mask[(int)(sy)*W + (int)(sx+1)];
            float c = mask[(int)(sy+1)*W + (int)(sx)];
            float d = mask[(int)(sy+1)*W + (int)(sx+1)];
            float x1 = a + (b-a)*xp; float x2 = c + (d-c)*xp;
            alpha = x1 + (x2-x1)*yp;
        }

        for(int i=0;i<3;i++){
            float c = color[i]*255*alpha + Xt[tid*3+(i)]*(1-alpha);
            c = c < 0 ? 0 : c;
            c = c > 255 ? 255 : c;
            ret[tid*3+i] = c;
        }

        tid += offset;
    }
}

void restore_image(uchar* ret, const float* Yst, const float* mask, const uchar* Xt,
        const float* trans, const int H, const int W, const int h, const int w) {
    restore_kernel<<<1000, 64>>>(ret, Yst, mask, Xt, trans, H, W, h, w);
    cudaThreadSynchronize();
}
