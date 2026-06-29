#pragma once

#include <nuTens/propagator/base-mixing-matrix.hpp>
#include <nuTens/tensors/tensor.hpp>

namespace nuTens
{

constexpr std::complex<float> imagUnit(0.0, 1.0);

/// @brief PMNS matrix in the standard parameterisation
/// Convenient way to construct the matrix
class PMNSmatrix : public BaseMixingMatrix
{
  public:
    PMNSmatrix(dtypes::deviceType device = dtypes::kCPU) : BaseMixingMatrix(device)
    {
        NT_PROFILE();

        // set up the three matrices to build the mixing matrix
        _mat1 = Tensor::zeros({1, 3, 3}, dtypes::kComplexFloat).requiresGrad(false).device(_device);
        _mat2 = Tensor::zeros({1, 3, 3}, dtypes::kComplexFloat).requiresGrad(false).device(_device);
        _mat3 = Tensor::zeros({1, 3, 3}, dtypes::kComplexFloat).requiresGrad(false).device(_device);
    }

    inline PMNSmatrix &setTheta12(float theta12)
    {
        NT_PROFILE();

        if (_device == dtypes::kCPU)
        {
            _theta12.requiresGrad(false);

            _theta12.setValue({0}, theta12);

            _theta12.requiresGrad(true);
        }

        else if (_device == dtypes::kGPU)
        {
            _theta12gpu.requiresGrad(false);

            _theta12gpu.setValue({0}, theta12);

            _theta12gpu.requiresGrad(true);
        }

        // set the dirty flag
        _needsRecalculating = true;

        return *this;
    }

    inline PMNSmatrix &setTheta13(float theta13)
    {
        NT_PROFILE();

        if (_device == dtypes::kCPU)
        {
            _theta13.requiresGrad(false);

            _theta13.setValue({0}, theta13);

            _theta13.requiresGrad(true);
        }
        else if (_device == dtypes::kGPU)
        {
            _theta13gpu.requiresGrad(false);

            _theta13gpu.setValue({0}, theta13);

            _theta13gpu.requiresGrad(true);
        }
        // set the dirty flag
        _needsRecalculating = true;

        return *this;
    }

    inline PMNSmatrix &setTheta23(float theta23)
    {
        NT_PROFILE();

        if (_device == dtypes::kCPU)
        {
            _theta23.requiresGrad(false);

            _theta23.setValue({0}, theta23);

            _theta23.requiresGrad(true);
        }
        else if (_device == dtypes::kGPU)
        {
            _theta23gpu.requiresGrad(false);

            _theta23gpu.setValue({0}, theta23);

            _theta23gpu.requiresGrad(true);
        }
        // set the dirty flag
        _needsRecalculating = true;

        return *this;
    }

    inline PMNSmatrix &setDeltaCP(float deltaCP)
    {
        NT_PROFILE();

        _deltaCP.requiresGrad(false);

        _deltaCP.setValue({0}, deltaCP);

        _deltaCP.requiresGrad(true);

        // set the dirty flag
        _needsRecalculating = true;

        return *this;
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

  protected:
    inline Tensor _build() override
    {
        NT_PROFILE();

        if (_device == dtypes::kCPU)
        {
            buildMat1(_theta23);
            buildMat2(_theta13);
            buildMat3(_theta12);
        }
        else if (_device == dtypes::kGPU)
        {
            buildMat1(_theta23gpu);
            buildMat2(_theta13gpu);
            buildMat3(_theta12gpu);
        }
        // Build PMNS
        return Tensor::matmul(_mat1, Tensor::matmul(_mat2, _mat3));
    }

  private:
    inline void buildMat1(const Tensor &theta23)
    {
        _mat1.setValue({0, 0, 0}, 1.0);
        _mat1.setValue({0, 1, 1}, Tensor::cos(theta23));
        _mat1.setValue({0, 1, 2}, Tensor::sin(theta23));
        _mat1.setValue({0, 2, 1}, -Tensor::sin(theta23));
        _mat1.setValue({0, 2, 2}, Tensor::cos(theta23));
    }
    inline void buildMat2(const Tensor &theta13)
    {
        _mat2.setValue({0, 1, 1}, 1.0);
        _mat2.setValue({0, 0, 0}, Tensor::cos(theta13));
        _mat2.setValue({0, 0, 2}, Tensor::mul(Tensor::sin(theta13), Tensor::exp(Tensor::scale(_deltaCP, -imagUnit))));
        _mat2.setValue({0, 2, 0}, -Tensor::mul(Tensor::sin(theta13), Tensor::exp(Tensor::scale(_deltaCP, imagUnit))));
        _mat2.setValue({0, 2, 2}, Tensor::cos(theta13));
    }
    inline void buildMat3(const Tensor &theta12)
    {
        _mat3.setValue({0, 2, 2}, 1.0);
        _mat3.setValue({0, 0, 0}, Tensor::cos(theta12));
        _mat3.setValue({0, 0, 1}, Tensor::sin(theta12));
        _mat3.setValue({0, 1, 0}, -Tensor::sin(theta12));
        _mat3.setValue({0, 1, 1}, Tensor::cos(theta12));
    }
    // the mixing parameters
    AccessedTensor<float, 1, dtypes::kCPU> _theta12 = AccessedTensor<float, 1, dtypes::kCPU>::zeros({1}, true);
    AccessedTensor<float, 1, dtypes::kCPU> _theta13 = AccessedTensor<float, 1, dtypes::kCPU>::zeros({1}, true);
    AccessedTensor<float, 1, dtypes::kCPU> _theta23 = AccessedTensor<float, 1, dtypes::kCPU>::zeros({1}, true);

    // keep gpu and cpu versions so we can switch at runtime
    Tensor _theta12gpu = Tensor::zeros({1}, dtypes::kFloat, dtypes::kGPU, true);
    Tensor _theta13gpu = Tensor::zeros({1}, dtypes::kFloat, dtypes::kGPU, true);
    Tensor _theta23gpu = Tensor::zeros({1}, dtypes::kFloat, dtypes::kGPU, true);

    Tensor _deltaCP = Tensor::zeros({1}, dtypes::kComplexFloat, _device, true);

    // the sub-matrices
    Tensor _mat1;
    Tensor _mat2;
    Tensor _mat3;
};

}; // namespace nuTens