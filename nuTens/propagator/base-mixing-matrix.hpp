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

        // batch dim missing
        if (_matrix.getNdim() == 2)
        {
            if (_batchSize != 1)
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

        if (_matrix.getShape()[0] != _batchSize)
        {
            NT_ERROR("Your mixing matrix class returned a mixing matrix tensor whose batch dim != the batch size");
            NT_ERROR("Expected ", _batchSize, " but got ", _matrix.getShape()[0]);
            throw std::runtime_error("Bad batching");
        }

        _matrix.hasBatchDim(true);

        return _matrix;
    };

    /// @brief Constructor
    BaseMixingMatrix(dtypes::deviceType device = dtypes::kCPU, long batchSize = 1)
        : _device(device), _batchSize(batchSize){};

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

    /// @brief get number of batches
    /// Value of 0 means the batch size has not yet been initialised
    [[nodiscard]] inline long getBatchSize() const
    {
        return _batchSize;
    };

  protected:
    /// @brief Should construct and return the mixing matrix
    virtual Tensor _build() = 0;

    /// flag to set if the matrix needs to be recalculated or if it's fine to
    /// just return the cached one
    bool _needsRecalculating = true;

    /// The number of batches of mixing matrix values
    long _batchSize;

    /// The device that this object lives on
    dtypes::deviceType _device;

    /// Cached mixing matrix
    Tensor _matrix;
};

}; // namespace nuTens
