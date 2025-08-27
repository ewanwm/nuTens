#pragma once

#include <nuTens/propagator/base-mixing-matrix.hpp>
#include <nuTens/tensors/tensor.hpp>

namespace nuTens
{

const std::complex<float> imagUnit(0.0, 1.0);

/// @brief PMNS matrix in the standard parameterisation
/// Convenient way to construct the matrix
class PMNSmatrix : public BaseMixingMatrix
{
  public:
    PMNSmatrix()
    {
        NT_PROFILE();

        // set up the three matrices to build the mixing matrix
        _mat1 = Tensor::zeros({1, 3, 3}, dtypes::kComplexFloat).requiresGrad(false);
        _mat2 = Tensor::zeros({1, 3, 3}, dtypes::kComplexFloat).requiresGrad(false);
        _mat3 = Tensor::zeros({1, 3, 3}, dtypes::kComplexFloat).requiresGrad(false);
    }

    inline Tensor &build() override
    {
        NT_PROFILE();

        _mat1.setValue({0, 0, 0}, 1.0);
        _mat1.setValue({0, 1, 1}, Tensor::cos(_theta23));
        _mat1.setValue({0, 1, 2}, Tensor::sin(_theta23));
        _mat1.setValue({0, 2, 1}, -Tensor::sin(_theta23));
        _mat1.setValue({0, 2, 2}, Tensor::cos(_theta23));
        _mat1.requiresGrad(true);

        _mat2.setValue({0, 1, 1}, 1.0);
        _mat2.setValue({0, 0, 0}, Tensor::cos(_theta13));
        _mat2.setValue({0, 0, 2}, Tensor::mul(Tensor::sin(_theta13), Tensor::exp(Tensor::scale(_deltaCP, -imagUnit))));
        _mat2.setValue({0, 2, 0}, -Tensor::mul(Tensor::sin(_theta13), Tensor::exp(Tensor::scale(_deltaCP, imagUnit))));
        _mat2.setValue({0, 2, 2}, Tensor::cos(_theta13));
        _mat2.requiresGrad(true);

        _mat3.setValue({0, 2, 2}, 1.0);
        _mat3.setValue({0, 0, 0}, Tensor::cos(_theta12));
        _mat3.setValue({0, 0, 1}, Tensor::sin(_theta12));
        _mat3.setValue({0, 1, 0}, -Tensor::sin(_theta12));
        _mat3.setValue({0, 1, 1}, Tensor::cos(_theta12));
        _mat3.requiresGrad(true);

        // Build PMNS
        _matrix = Tensor::matmul(_mat1, Tensor::matmul(_mat2, _mat3));
        return _matrix;
    }

    inline void setParameterValues(float theta12, float theta13, float theta23, float deltaCP)
    {
        NT_PROFILE();

        _theta12.requiresGrad(false);
        _theta13.requiresGrad(false);
        _theta23.requiresGrad(false);
        _deltaCP.requiresGrad(false);

        _theta12.setValue(theta12, 0);
        _theta13.setValue(theta13, 0);
        _theta23.setValue(theta23, 0);
        _deltaCP.setValue({0}, deltaCP);

        _theta12.requiresGrad(true);
        _theta13.requiresGrad(true);
        _theta23.requiresGrad(true);
        _deltaCP.requiresGrad(true);
    }

    /// @{Setters
    inline const Tensor &getTheta12Tensor()
    {
        return _theta12;
    }
    inline const Tensor &getTheta13Tensor()
    {
        return _theta13;
    }
    inline const Tensor &getTheta23Tensor()
    {
        return _theta23;
    }
    inline const Tensor &getDeltaCPTensor()
    {
        return _deltaCP;
    }
    /// @}

  private:
    // the mixing parameters
    AccessedTensor<float, 1, dtypes::kCPU> _theta12 = AccessedTensor<float, 1, dtypes::kCPU>::zeros({1}, true);
    AccessedTensor<float, 1, dtypes::kCPU> _theta13 = AccessedTensor<float, 1, dtypes::kCPU>::zeros({1}, true);
    AccessedTensor<float, 1, dtypes::kCPU> _theta23 = AccessedTensor<float, 1, dtypes::kCPU>::zeros({1}, true);
    Tensor _deltaCP = Tensor::zeros({1}, dtypes::kComplexFloat, dtypes::kCPU, true);

    // the sub-matrices
    Tensor _mat1;
    Tensor _mat2;
    Tensor _mat3;

    // the actual matrix
    Tensor _matrix;
};

}; // namespace nuTens