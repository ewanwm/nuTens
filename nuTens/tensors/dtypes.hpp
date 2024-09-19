#pragma once

#if USE_PYTORCH
#include <torch/torch.h>
#endif

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
    kUninitScalar,
};

/// Devices that a Tensor can live on
enum deviceType
{
    kCPU,
    kGPU,
    kUninitDevice,
};

#if USE_PYTORCH
/// map between the data types used in nuTens and those used by pytorch
const static std::map<scalarType, c10::ScalarType> scalarTypeMap = {{kFloat, torch::kFloat},
                                                                    {kDouble, torch::kDouble},
                                                                    {kComplexFloat, torch::kComplexFloat},
                                                                    {kComplexDouble, torch::kComplexDouble}};

/// inverse map between the data types used in nuTens and those used by pytorch
const static std::map<c10::ScalarType, scalarType> invScalarTypeMap = {{torch::kFloat, kFloat},
                                                                       {torch::kDouble, kDouble},
                                                                       {torch::kComplexFloat, kComplexFloat},
                                                                       {torch::kComplexDouble, kComplexDouble}};

// map between the device types used in nuTens and those used by pytorch
const static std::map<deviceType, c10::DeviceType> deviceTypeMap = {{kCPU, torch::kCPU}, {kGPU, torch::kCUDA}};

// inverse map between the device types used in nuTens and those used by pytorch
const static std::map<c10::DeviceType, deviceType> invDeviceTypeMap = {{torch::kCPU, kCPU}, {torch::kCUDA, kGPU}};
#endif
} // namespace NTdtypes