#pragma once

#include <nuTens/tensors/tensor.hpp>

namespace nuTens
{

class BaseMixingMatrix
{
  public:
    /// @brief Get the mixing matrix
    virtual Tensor &build()
    {
        if (_needsRecalculating)
        {
            _matrix = _build();

            // clear the dirty flag
            _needsRecalculating = false;
        }

        return _matrix;
    };

    /// @brief Constructor
    BaseMixingMatrix(dtypes::deviceType device = dtypes::kCPU) : _device(device){};

    /// @brief Destructor
    virtual ~BaseMixingMatrix() = default;
    /// @brief copy constructor
    BaseMixingMatrix(BaseMixingMatrix const &) = default;
    /// @brief copy assignment operator
    BaseMixingMatrix &operator=(BaseMixingMatrix const &) = default;
    /// @brief move constructor
    BaseMixingMatrix(BaseMixingMatrix &&) = default;
    /// @brief move assignment operator
    BaseMixingMatrix &operator=(BaseMixingMatrix &&) = default;

  protected:
    /// @brief Should construct and return the mixing matrix
    virtual Tensor _build() = 0;

    /// flag to set if the matrix needs to be recalculated or if it's fine to
    /// just return the cached one
    bool _needsRecalculating = true;

    /// The device that this object lives on
    dtypes::deviceType _device;

    /// Cached mixing matrix
    Tensor _matrix;
};

}; // namespace nuTens
