#pragma once

#include <utility>

#include <nuTens/tensors/dtypes.hpp>
#include <nuTens/tensors/tensor.hpp>
#include <nuTens/utils/instrumentation.hpp>
#include <nuTens/utils/logging.hpp>

namespace nuTens
{

class ModuleBase
{
    /*!
     * @class ModuleBase
     * @brief Base class for any "module" in nuTens - anything that does calculations
     *
     * Modules are anything that performs operations on tensors in the process of
     * calculating oscillation probabilities.
     *
     * If you want to create a custom module then it should inherit from this base
     * class. It provides a number of helpful utilities such as validating parameters.
     */

  public:
    ModuleBase(long batchSize, dtypes::deviceType device, std::string name)
        : _batchSize(batchSize), _device(device), _name(std::move(name))
    {
    }

    /// @brief Get the name of this module
    [[nodiscard]] const inline std::string &getName() const
    {
        return _name;
    }

    /// @brief get number of batches
    [[nodiscard]] inline long getBatchSize() const
    {
        return _batchSize;
    };

    /// @brief get number of batches
    [[nodiscard]] inline dtypes::deviceType getDevice() const
    {
        return _device;
    };

  protected:
    /// @brief Set the name of this module
    /// @param newName
    inline void setName(const std::string &newName)
    {
        _name = newName;
    }

    /// @brief Performs a check that the provided parameter has the expected shape and is batched according to
    /// expectation from this modules _batchSize attribute
    /// @param[in] parameter The parameter to be checked
    /// @param[in] expectNdims The number of expected dimensions (excluding the batch dimension)
    /// @param[in] expectShape The expected shape (excluding the batch dimension) - The size of this must match
    /// expectNdimes
    /// @param[out] batchedParamRet Reference to the provided input parameter with added batch dimension if required -
    /// e.g. if the input parameter did not have a batch dimension already, and _batchSize is 1 this will be `parameter`
    /// with `unsqueeze(0)` applied This will emit some `NT_ERROR`s to inform the user what the problems are with the
    /// shape if any
    inline void checkParameterShape(const Tensor &parameter, long expectNdims, const std::vector<long> &expectShape,
                                    Tensor &batchedParamRet, const std::string &parameterName) const
    {

        NT_PROFILE();

        bool valid = true;

        // get expected shape as a string
        std::string shapeString;
        for (const long &dimSize : expectShape)
        {
            shapeString += std::to_string(dimSize) + ", ";
        }
        // get actual shape as a string
        std::string actualShapeString;
        const auto actualPreRegShape = parameter.getShape();
        for (const long &dimSize : actualPreRegShape)
        {
            actualShapeString += std::to_string(dimSize) + ", ";
        }

        if (parameter.getNdim() == expectNdims)
        {
            if (_batchSize != 1)
            {
                NT_ERROR("Parameter without batch dimension is only supported if batch size is 1!");
                valid = false;
            }

            batchedParamRet = parameter.unsqueeze(0);
        }
        else if (parameter.getNdim() == expectNdims + 1)
        {
            if (parameter.getShape()[0] != _batchSize)
            {
                NT_ERROR("first dimension of parameter tensor does not match the batch size!");
                valid = false;
            }

            batchedParamRet = parameter;
        }
        else
        {
            NT_ERROR("Parameter has invalid number of dimensions!");

            valid = false;
        }

        // if valid is still true, we at least have the right number of dimensions so can check the shape
        if (valid)
        {
            // batchedParamRet definitely has batch dim, so we know it is the right format
            // so we use that for shape checking
            const auto actualShape = batchedParamRet.getShape();

            // loop skipping batch dim, which has been added if needed above
            for (int dim = 0; dim < expectNdims; dim++)
            {
                if (actualShape[dim + 1] != expectShape[dim])
                {
                    std::cout << actualShape[dim] << " != " << expectShape[dim] << std::endl;
                    valid = false;
                }
            }
            if (!valid)
            {
                NT_ERROR("Parameter has bad shape");
            }
        }

        if (!valid)
        {
            NT_ERROR("Parameter {} provided to module {} must have either:", parameterName, getName());
            NT_ERROR("  {} dimension(s): [{}] - only valid if the batch size of the propagator is 1", expectNdims,
                     shapeString);
            NT_ERROR("  {} dimension(s): [n_batches ({}), {}]", 1 + expectNdims, _batchSize, shapeString);
            NT_ERROR("Got actual shape: [{}]", actualShapeString);
            throw std::invalid_argument("Bad parameter shape");
        }
    }

  private:
    std::string _name;
    long _batchSize;
    dtypes::deviceType _device;
};

} // end namespace nuTens