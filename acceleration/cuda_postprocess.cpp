#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <opencv2/opencv.hpp>
#include "cuda_kernels.hpp"

using std::cout;
using std::endl;

namespace py = pybind11;
using namespace std;

class CudaPostprocess{
public:
    CudaPostprocess(int H, int W){
        this->H = H;
        this->W = W;
        cudaMalloc((void**)&Yst_dev, H*W*3*sizeof(float));
        cudaMalloc((void**)&mask_dev, H*W*sizeof(float));
        cudaMalloc((void**)&trans_dev, 6*sizeof(float));
        ret_dev = NULL;
        Xt_raw_dev = NULL;
        ret_size = 0;
    }
    py::array_t<uchar> restore(py::array_t<float, py::array::c_style|py::array::forcecast> Yst, py::array_t<float, py::array::c_style|py::array::forcecast> mask,
            py::array_t<float, py::array::c_style|py::array::forcecast> trans, py::array_t<uchar, py::array::c_style|py::array::forcecast> Xt_raw, int h, int w) {
        py::buffer_info Yst_buf = Yst.request();
        float* Yst_ptr = (float*)Yst_buf.ptr;

        py::buffer_info mask_buf = mask.request();
        float* mask_ptr = (float*)mask_buf.ptr;

        py::buffer_info Xt_buf = Xt_raw.request();
        uchar* Xt_ptr = (uchar*)Xt_buf.ptr;

        float* trans_ptr = (float*)trans.request().ptr;

        if (h * w * 3 * sizeof(uchar) > ret_size){
            if (ret_dev != NULL){
                cudaFree(ret_dev);
                cudaFree(Xt_raw_dev);
                ret_dev = NULL;
                Xt_raw_dev = NULL;
            }
            cudaMalloc((void**)&ret_dev, h*w*3*sizeof(uchar));
            cudaMalloc((void**)&Xt_raw_dev, h*w*3*sizeof(uchar));
            ret_size = h * w * 3 * sizeof(uchar);
        }
        cudaMemcpy(Yst_dev, Yst_ptr, H*W*3*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(mask_dev, mask_ptr, H*W*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(trans_dev, trans_ptr, 6*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(Xt_raw_dev, Xt_ptr, h*w*3*sizeof(uchar), cudaMemcpyHostToDevice);

        restore_image(ret_dev, Yst_dev, mask_dev, Xt_raw_dev, trans_dev, H, W, h, w);

        auto ret = py::array_t<uchar>({h, w, 3});
        py::buffer_info info = ret.request();
        cudaMemcpy(info.ptr, ret_dev, h*w*3*sizeof(uchar), cudaMemcpyDeviceToHost);
        return ret;
    }
    ~CudaPostprocess(){
        cudaFree(Yst_dev);
        cudaFree(mask_dev);
        cudaFree(trans_dev);
        if (ret_dev != NULL)
            cudaFree(ret_dev);
    }

private:
    float *Yst_dev, *mask_dev;
    uchar *ret_dev;
    uchar *Xt_raw_dev;
    int ret_size;
    float *trans_dev;
    int H, W;
};

PYBIND11_MODULE(cuda_postprocess, m)
{
    m.doc() = "FaceShifter postprocess accelerated by cuda";
    py::class_<CudaPostprocess>(m, "CudaPostprocess")
        .def(py::init<int, int>())
        .def("restore", &CudaPostprocess::restore);
}




/* EOF */

