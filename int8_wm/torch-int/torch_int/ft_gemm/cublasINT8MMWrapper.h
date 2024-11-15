/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "cuda_utils.h"
// #include "src/fastertransformer/layers/attention_layers/AttentionWeight.h"
#include "cublasAlgoMap.h"
#include "cublasMMWrapper.h"
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <map>
#include <mutex>
#include <string>
#include "pybind11/pybind11.h"

#pragma once
 

class cublasINT8MMWrapper: public cublasMMWrapper {
private:
    bool use_ORDER_COL32_2R_4R4_;

public:
    cublasINT8MMWrapper(cublasLtHandle_t cublaslt_handle_,
                        cudaStream_t     stream,
                        cublasAlgoMap*   map,
                        std::mutex*      mu,
                        bool             use_ORDER_COL32_2R_4R4);

    cublasINT8MMWrapper(cublasHandle_t   cublas_handle,
                        cublasLtHandle_t cublaslt_handle,
                        cudaStream_t     stream,
                        cublasAlgoMap*   map,
                        std::mutex*      mu,
                        bool             use_ORDER_COL32_2R_4R4);

    ~cublasINT8MMWrapper();

    cublasINT8MMWrapper(const cublasINT8MMWrapper& wrapper);

    void Gemm(int*          res,
              int           batchCount,
              int           m,
              int           n,
              int           k,
              int64_t       stridea,
              int64_t       strideb,
              int64_t       stridec,
              const int8_t* ATransform,
              const int8_t* kernel);

    void Gemm(int8_t*       res,
              int           batchCount,
              int           m,
              int           n,
              int           k,
              int64_t       stridea,
              int64_t       strideb,
              int64_t       stridec,
              const float   alpha,
              const int8_t* ATransform,
              const int8_t* kernel);

    // template<typename T>
    // int getFusedINT8QKVType(const int k, const int n, const AttentionWeight<T>* attention_weights);

    bool getUseOrderCol322R4R4();
};

 

// PYBIND11_MODULE(FTGEMM, m) {
//     pybind11::class_<cublasINT8MMWrapper>(m, "cublasINT8MMWrapper")
//         .def(pybind11::init<const cublasINT8MMWrapper&>())
//         .def(pybind11::init<cublasLtHandle_t, cudaStream_t, cublasAlgoMap*, std::mutex*, bool>())
//         .def(pybind11::init<cublasHandle_t, cublasLtHandle_t, cudaStream_t, cublasAlgoMap*, std::mutex*, bool>())
//         .def("cublasINT8MMWrapper",  (void (cublasINT8MMWrapper::*)(int*, int, int, int, int, int64_t, int64_t, int64_t, const int8_t*, const int8_t*)) &cublasINT8MMWrapper::Gemm)
//         .def("cublasINT8MMWrapper",  (void (cublasINT8MMWrapper::*)(int8_t*, int, int, int, int, int64_t, int64_t, int64_t, const float,const int8_t*, const int8_t*)) &cublasINT8MMWrapper::Gemm)
//         .def("getUseOrderCol322R4R4", &cublasINT8MMWrapper::getUseOrderCol322R4R4);
// }