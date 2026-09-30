#pragma once

#include <nuTens/propagator/module-base.hpp>
#include <nuTens/tensors/tensor.hpp>

namespace nuTens
{

class BaseMixingMatrix : public ModuleBase
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

        // batch dim missing
        if (_matrix.getNdim() == 2)
        {
            if (getBatchSize() != 1)
            {
                NT_ERROR("Your Mixing matrix class can only return a 2D tensor if batch size == 1");
                throw std::runtime_error("Bad batching");
            }
            _matrix.addBatchDim();
        }

        if (_matrix.getNdim() != 3)
        {
            NT_ERROR("Wrong number of dimensions in mixing matrix!");
            NT_ERROR("  Your mixing matrix class should provide a _build() method that returns a tensor with either");
            NT_ERROR("  3 dimensions: (nBatches, nRows, nColumns)");
            NT_ERROR("  2 dimensions: (nRows, nColumns) - only allowed if batch size is 1");
            throw std::runtime_error("Bad mixing matrix");
        }

        if (_matrix.getShape()[0] != getBatchSize())
        {
            NT_ERROR("Your mixing matrix class returned a mixing matrix tensor whose batch dim != the batch size");
            NT_ERROR("Expected ", getBatchSize(), " but got ", _matrix.getShape()[0]);
            throw std::runtime_error("Bad batching");
        }

        _matrix.hasBatchDim(true);

        return _matrix;
    };

    /// @brief Constructor
    BaseMixingMatrix(dtypes::deviceType device = dtypes::kCPU, long batchSize = 1)
        : ModuleBase(batchSize, device, "BaseMixingMatrix"){};

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

    /// Cached mixing matrix
    Tensor _matrix;
};

}; // namespace nuTens
