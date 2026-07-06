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

        if (device == dtypes::kCPU)
        {
            _theta12 = std::make_shared<Tensor>(AccessedTensor<float, 1, dtypes::kCPU>::zeros({1}, true));
            _theta13 = std::make_shared<Tensor>(AccessedTensor<float, 1, dtypes::kCPU>::zeros({1}, true));
            _theta23 = std::make_shared<Tensor>(AccessedTensor<float, 1, dtypes::kCPU>::zeros({1}, true));
        }
        else if (device == dtypes::kGPU)
        {
            _theta12 =
                std::make_shared<Tensor>(Tensor::zeros({1}).dType(dtypes::kFloat).device(_device).requiresGrad(true));
            _theta13 =
                std::make_shared<Tensor>(Tensor::zeros({1}).dType(dtypes::kFloat).device(_device).requiresGrad(true));
            _theta23 =
                std::make_shared<Tensor>(Tensor::zeros({1}).dType(dtypes::kFloat).device(_device).requiresGrad(true));
        }

        // set up the three matrices to build the mixing matrix
        _mat1 = Tensor::zeros({1, 3, 3}, dtypes::kComplexFloat).requiresGrad(false).device(_device);
        _mat2 = Tensor::zeros({1, 3, 3}, dtypes::kComplexFloat).requiresGrad(false).device(_device);
        _mat3 = Tensor::zeros({1, 3, 3}, dtypes::kComplexFloat).requiresGrad(false).device(_device);
    }

    inline PMNSmatrix &setTheta12(float theta12)
    {
        NT_PROFILE();

        _theta12->requiresGrad(false);

        _theta12->setValue({0}, theta12);

        _theta12->requiresGrad(true);

        // set the dirty flag
        _needsRecalculating = true;

        return *this;
    }

    inline PMNSmatrix &setTheta13(float theta13)
    {
        NT_PROFILE();

        _theta13->requiresGrad(false);

        _theta13->setValue({0}, theta13);

        _theta13->requiresGrad(true);

        // set the dirty flag
        _needsRecalculating = true;

        return *this;
    }

    inline PMNSmatrix &setTheta23(float theta23)
    {
        NT_PROFILE();

        _theta23->requiresGrad(false);

        _theta23->setValue({0}, theta23);

        _theta23->requiresGrad(true);

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
        return *_theta12;
    }
    inline const Tensor &getTheta13Tensor()
    {
        return *_theta13;
    }
    inline const Tensor &getTheta23Tensor()
    {
        return *_theta23;
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

        buildMat1();
        buildMat2();
        buildMat3();

        // Build PMNS
        return Tensor::matmul(_mat1, Tensor::matmul(_mat2, _mat3));
    }

  private:
    inline void buildMat1()
    {
        _mat1.setValue({0, 0, 0}, 1.0);
        _mat1.setValue({0, 1, 1}, Tensor::cos(*_theta23));
        _mat1.setValue({0, 1, 2}, Tensor::sin(*_theta23));
        _mat1.setValue({0, 2, 1}, -Tensor::sin(*_theta23));
        _mat1.setValue({0, 2, 2}, Tensor::cos(*_theta23));
    }
    inline void buildMat2()
    {
        _mat2.setValue({0, 1, 1}, 1.0);
        _mat2.setValue({0, 0, 0}, Tensor::cos(*_theta13));
        _mat2.setValue({0, 0, 2}, Tensor::mul(Tensor::sin(*_theta13), Tensor::exp(Tensor::scale(_deltaCP, -imagUnit))));
        _mat2.setValue({0, 2, 0}, -Tensor::mul(Tensor::sin(*_theta13), Tensor::exp(Tensor::scale(_deltaCP, imagUnit))));
        _mat2.setValue({0, 2, 2}, Tensor::cos(*_theta13));
    }
    inline void buildMat3()
    {
        _mat3.setValue({0, 2, 2}, 1.0);
        _mat3.setValue({0, 0, 0}, Tensor::cos(*_theta12));
        _mat3.setValue({0, 0, 1}, Tensor::sin(*_theta12));
        _mat3.setValue({0, 1, 0}, -Tensor::sin(*_theta12));
        _mat3.setValue({0, 1, 1}, Tensor::cos(*_theta12));
    }
    // the mixing parameters
    std::shared_ptr<Tensor> _theta12;
    std::shared_ptr<Tensor> _theta13;
    std::shared_ptr<Tensor> _theta23;

    Tensor _deltaCP = Tensor::zeros({1}, dtypes::kComplexFloat, _device, true);

    // the sub-matrices
    Tensor _mat1;
    Tensor _mat2;
    Tensor _mat3;
};

}; // namespace nuTens