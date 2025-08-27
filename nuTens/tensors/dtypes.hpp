#pragma once

#if USE_PYTORCH
#include <torch/torch.h>
#endif

/*!
 * @file dtypes.hpp
 * @brief Defines various datatypes used in the project
 */

namespace nuTens
{

namespace dtypes
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

/// map between raw c++ types and the data types used in nuTens
template <typename T> static constexpr scalarType scalarTypeFromRaw();

template <> constexpr scalarType scalarTypeFromRaw<float>()
{
    return kFloat;
}
template <> constexpr scalarType scalarTypeFromRaw<double>()
{
    return kDouble;
}
template <> constexpr scalarType scalarTypeFromRaw<std::complex<float>>()
{
    return kComplexFloat;
}
template <> constexpr scalarType scalarTypeFromRaw<std::complex<double>>()
{
    return kComplexDouble;
}

#if USE_PYTORCH

/// map between the data types used in nuTens and those used by pytorch
constexpr std::pair<scalarType, c10::ScalarType> scalarTypeMapVals[] = {{kFloat, torch::kFloat},
                                                                        {kDouble, torch::kDouble},
                                                                        {kComplexFloat, torch::kComplexFloat},
                                                                        {kComplexDouble, torch::kComplexDouble}};

constexpr auto scalarTypeMapSize = sizeof scalarTypeMapVals / sizeof scalarTypeMapVals[0];

static constexpr c10::ScalarType scalarTypeMap(scalarType key, int range = scalarTypeMapSize)
{
    return (range == 0)                                  ? throw "Key not present"
           : (scalarTypeMapVals[range - 1].first == key) ? scalarTypeMapVals[range - 1].second
                                                         : scalarTypeMap(key, range - 1);
};

static constexpr scalarType invScalarTypeMap(c10::ScalarType value, int range = scalarTypeMapSize)
{
    return (range == 0)                                     ? throw "Value not present"
           : (scalarTypeMapVals[range - 1].second == value) ? scalarTypeMapVals[range - 1].first
                                                            : invScalarTypeMap(value, range - 1);
};

static_assert(invScalarTypeMap(scalarTypeMap(kComplexFloat)) == kComplexFloat, "should be inverse");

// map between the device types used in nuTens and those used by pytorch
constexpr std::pair<deviceType, c10::DeviceType> deviceTypeMapVals[] = {{kCPU, torch::kCPU}, {kGPU, torch::kCUDA}};

constexpr auto deviceTypeMapSize = sizeof deviceTypeMapVals / sizeof deviceTypeMapVals[0];

static constexpr c10::DeviceType deviceTypeMap(deviceType key, int range = deviceTypeMapSize)
{
    return (range == 0)                                  ? throw "Key not present"
           : (deviceTypeMapVals[range - 1].first == key) ? deviceTypeMapVals[range - 1].second
                                                         : deviceTypeMap(key, range - 1);
};

static constexpr deviceType invDeviceTypeMap(c10::DeviceType value, int range = deviceTypeMapSize)
{
    return (range == 0)                                     ? throw "Value not present"
           : (deviceTypeMapVals[range - 1].second == value) ? deviceTypeMapVals[range - 1].first
                                                            : invDeviceTypeMap(value, range - 1);
};

static_assert(invDeviceTypeMap(deviceTypeMap(kCPU)) == kCPU, "should be inverse");

#endif
} // namespace dtypes
} // namespace nuTens