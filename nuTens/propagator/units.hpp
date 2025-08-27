#pragma once

/// @file units.hpp
/// @brief Defines some suuuper basic units to convert to eV

// 1 eV^-1 = 1.239841984 e-6 m
// => 1m = 0.806554394 e^6 eV^-1

namespace nuTens
{

namespace units
{

static constexpr double eV = 1.0;  // eV
static constexpr double MeV = 1e6; // eV
static constexpr double GeV = 1e9; // eV

static constexpr double m = 0.806554394e6; // eV^-1
static constexpr double cm = 1e-2 * m;     // eV^-1
static constexpr double km = 1e3 * m;      // eV^-1

} // namespace units

} // namespace nuTens
