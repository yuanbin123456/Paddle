/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/math/math_function.h"

#ifdef PADDLE_WITH_MKLML
#include "paddle/fluid/platform/dynload/mklml.h"
#endif

#ifdef PADDLE_USE_OPENBLAS
#include <cblas.h>
#endif

#include <memory>
#include <utility>
#include <vector>
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/operators/math/math_function_impl.h"
#include "paddle/fluid/platform/bfloat16.h"
#include "paddle/fluid/platform/float16.h"
#include "unsupported/Eigen/CXX11/Tensor"

namespace paddle {
namespace operators {
namespace math {

using float16 = paddle::platform::float16;

template struct SetConstant<platform::SWDeviceContext, platform::float16>;
template struct SetConstant<platform::SWDeviceContext, platform::bfloat16>;
template struct SetConstant<platform::SWDeviceContext, float>;
template struct SetConstant<platform::SWDeviceContext, double>;
template struct SetConstant<platform::SWDeviceContext, int>;
template struct SetConstant<platform::SWDeviceContext, int64_t>;
template struct SetConstant<platform::SWDeviceContext, bool>;
template struct SetConstant<platform::SWDeviceContext, uint8_t>;
template struct SetConstant<platform::SWDeviceContext, platform::complex64>;
template struct SetConstant<platform::SWDeviceContext, platform::complex128>;

#ifdef PADDLE_WITH_XPU
template struct SetConstant<platform::XPUDeviceContext, platform::float16>;
template struct SetConstant<platform::XPUDeviceContext, platform::bfloat16>;
template struct SetConstant<platform::XPUDeviceContext, float>;
template struct SetConstant<platform::XPUDeviceContext, double>;
template struct SetConstant<platform::XPUDeviceContext, uint8_t>;
template struct SetConstant<platform::XPUDeviceContext, int>;
template struct SetConstant<platform::XPUDeviceContext, int64_t>;
template struct SetConstant<platform::XPUDeviceContext, bool>;
template struct SetConstant<platform::XPUDeviceContext, platform::complex64>;
template struct SetConstant<platform::XPUDeviceContext, platform::complex128>;
#endif

#define DEFINE_SW_TRANS(RANK)                                                \
  template struct Transpose<platform::SWDeviceContext, platform::float16,    \
                            RANK>;                                            \
  template struct Transpose<platform::SWDeviceContext, platform::bfloat16,   \
                            RANK>;                                            \
  template struct Transpose<platform::SWDeviceContext, float, RANK>;         \
  template struct Transpose<platform::SWDeviceContext, double, RANK>;        \
  template struct Transpose<platform::SWDeviceContext, int, RANK>;           \
  template struct Transpose<platform::SWDeviceContext, int64_t, RANK>;       \
  template struct Transpose<platform::SWDeviceContext, bool, RANK>;          \
  template struct Transpose<platform::SWDeviceContext, int16_t, RANK>;       \
  template struct Transpose<platform::SWDeviceContext, uint8_t, RANK>;       \
  template struct Transpose<platform::SWDeviceContext, int8_t, RANK>;        \
  template struct Transpose<platform::SWDeviceContext, platform::complex64,  \
                            RANK>;                                            \
  template struct Transpose<platform::SWDeviceContext, platform::complex128, \
                            RANK>;

DEFINE_SW_TRANS(1);
DEFINE_SW_TRANS(2);
DEFINE_SW_TRANS(3);
DEFINE_SW_TRANS(4);
DEFINE_SW_TRANS(5);
DEFINE_SW_TRANS(6);

template <typename T>
struct TransposeNormal<platform::SWDeviceContext, T> {
  void operator()(const platform::SWDeviceContext& context,
                  const framework::Tensor& in, framework::Tensor* out,
                  const std::vector<int>& axis) {
    const int rank = axis.size();
    auto in_stride = framework::stride(in.dims());
    auto out_stride = framework::stride(out->dims());
    const T* in_ptr = in.data<T>();
    T* out_ptr = out->data<T>();

    auto transpose_helper = [&](int64_t beg, int64_t end) {
      for (int64_t out_idx = beg; out_idx < end; ++out_idx) {
        int64_t in_idx = 0;
        int64_t tmp_idx = out_idx;
        // calculate the input index
        for (int i = 0; i < rank; ++i) {
          const int64_t coordinate = tmp_idx / out_stride[i];
          tmp_idx -= coordinate * out_stride[i];
          in_idx += coordinate * in_stride[axis[i]];
        }
        out_ptr[out_idx] = in_ptr[in_idx];
      }
    };
    transpose_helper(0, out->numel());
  }
};

// define transpose normal
#define DEFINE_SW_TRANS_NORMAL(TYPE) \
  template struct TransposeNormal<platform::SWDeviceContext, TYPE>

DEFINE_SW_TRANS_NORMAL(platform::float16);
DEFINE_SW_TRANS_NORMAL(platform::bfloat16);
DEFINE_SW_TRANS_NORMAL(float);
DEFINE_SW_TRANS_NORMAL(double);
DEFINE_SW_TRANS_NORMAL(int);
DEFINE_SW_TRANS_NORMAL(int64_t);
DEFINE_SW_TRANS_NORMAL(bool);
DEFINE_SW_TRANS_NORMAL(int16_t);
DEFINE_SW_TRANS_NORMAL(uint8_t);
DEFINE_SW_TRANS_NORMAL(int8_t);
DEFINE_SW_TRANS_NORMAL(platform::complex64);
DEFINE_SW_TRANS_NORMAL(platform::complex128);

struct TensorSetConstantSW {
  TensorSetConstantSW(framework::Tensor* tensor, float value)
      : tensor_(tensor), value_(value) {}
  template <typename T>
  void apply() const {
    auto cpu = platform::SWPlace();
    auto* begin = tensor_->mutable_data<T>(cpu);
    std::fill(begin, begin + tensor_->numel(), static_cast<T>(value_));
  }
  framework::Tensor* tensor_;
  float value_;
};

template <>
void set_constant_with_place<platform::XPUPlace>(
    const platform::DeviceContext& context, framework::Tensor* tensor,
    float value) {
  PADDLE_THROW(platform::errors::Unimplemented("XPUPlace is not supported"));
}

template <>
void set_constant_with_place<platform::NPUPlace>(
    const platform::DeviceContext& context, framework::Tensor* tensor,
    float value) {
  PADDLE_THROW(platform::errors::Unimplemented("NPUPlace is not supported"));
}

template <>
void set_constant_with_place<platform::SWPlace>(
    const platform::DeviceContext& context, framework::Tensor* tensor,
    float value) {
  framework::VisitDataType(tensor->type(), TensorSetConstantSW(tensor, value));
}

template <>
void set_constant_with_place<platform::CUDAPinnedPlace>(
    const platform::DeviceContext& context, framework::Tensor* tensor,
    float value) {
  framework::VisitDataType(tensor->type(), TensorSetConstantSW(tensor, value));
}

struct TensorSetConstantWithPlace : public boost::static_visitor<void> {
  TensorSetConstantWithPlace(const platform::DeviceContext& context,
                             framework::Tensor* tensor, float value)
      : context_(context), tensor_(tensor), value_(value) {}

  template <typename Place>
  void operator()(Place place) const {
    set_constant_with_place<Place>(context_, tensor_, value_);
  }

  const platform::DeviceContext& context_;
  framework::Tensor* tensor_;
  float value_;
};

void set_constant(const platform::DeviceContext& context,
                  framework::Tensor* tensor, float value) {
  TensorSetConstantWithPlace func(context, tensor, value);
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  tensor->place().apply_visitor(func);
#else
  func(platform::SWPlace());
#endif
}

template <typename T>
struct RowwiseAdd<platform::SWDeviceContext, T> {
  void operator()(const platform::SWDeviceContext& context,
                  const framework::Tensor& input,
                  const framework::Tensor& vector, framework::Tensor* output) {
    auto in_dims = input.dims();
    auto out_dims = output->dims();
    auto size = input.numel() / in_dims[0];
    PADDLE_ENFORCE_EQ(
        vector.numel(), size,
        platform::errors::InvalidArgument(
            "The input vector size"
            " should be equal to the size of each row of input tensor."
            " Expected vector size=%d, but received %d",
            size, vector.numel()));
    const char* in_dims_cstr = in_dims.to_str().c_str();
    const char* out_dims_cstr = out_dims.to_str().c_str();
    PADDLE_ENFORCE_EQ(out_dims, in_dims,
                      platform::errors::InvalidArgument(
                          "The output tensor shape should be same as the input"
                          " tensor shape. Expected output tensor shape: %s,"
                          " but received %s",
                          in_dims_cstr, out_dims_cstr));

    auto in = framework::EigenMatrix<T>::From(input);
    auto vec = framework::EigenVector<T>::Flatten(vector);
    auto out = framework::EigenMatrix<T>::From(*output);

    for (int64_t i = 0; i < in_dims[0]; ++i) {
      out.chip(i, 0) = in.chip(i, 0) + vec;
    }
  }
};

template struct RowwiseAdd<platform::SWDeviceContext, float>;
template struct RowwiseAdd<platform::SWDeviceContext, double>;

template struct ColwiseSum<platform::SWDeviceContext, float>;
template struct ColwiseSum<platform::SWDeviceContext, double>;
template struct ColwiseSum<platform::SWDeviceContext, int>;
template struct ColwiseSum<platform::SWDeviceContext, int64_t>;

template struct RowwiseSum<platform::SWDeviceContext, float>;
template struct RowwiseSum<platform::SWDeviceContext, double>;

template struct RowwiseMean<platform::SWDeviceContext, float>;
template struct RowwiseMean<platform::SWDeviceContext, double>;

template <typename T>
struct ElementwiseAddTo<platform::SWDeviceContext, T> {
  void operator()(platform::SWDeviceContext* ctx, const framework::Tensor& src,
                  framework::Tensor* dst) {
    auto in = framework::EigenVector<T>::Flatten(src);
    auto out = framework::EigenVector<T>::Flatten(*dst);
    auto& place = *(ctx->eigen_device());
    out.device(place) = out + in;
  }
};

/* template <typename T>
struct ElementwiseAddTo<platform::SWDeviceContext, T> {
  void operator()(platform::SWDeviceContext* ctx, const framework::Tensor& src,
                  framework::Tensor* dst) {
    auto in = framework::EigenVector<T>::Flatten(src);
    auto out = framework::EigenVector<T>::Flatten(*dst);
    auto& place = *(ctx->eigen_device());
    out.device(place) = out + in;
  }
};  */



}  // namespace math
}  // namespace operators
}  // namespace paddle
