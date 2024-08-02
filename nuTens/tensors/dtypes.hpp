#pragma once

/*!
 * @file dtypes.hpp
 * @brief Defines various datatypes used in the project
 */

namespace NTdtypes
{

/// Types of scalar values
enum scalarType
{
    kInt,
    kFloat,
    kDouble,
    kComplexFloat,
    kComplexDouble,
};

/// Devices that a Tensor can live on
enum deviceType
{
    kCPU,
    kGPU
};

} // namespace NTdtypes