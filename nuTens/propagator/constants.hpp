#pragma once

/// @file constants.hpp
/// @brief Defines constants to be used across the project

#include <nuTens/propagator/units.hpp>

namespace nuTens
{

namespace constants
{

static constexpr double Groot2 =
    0.76294e-4 * (units::eV * units::eV) /
    units::GeV; //!< sqrt(2)*G_fermi in (eV^2-cm^3)/(mole-GeV) used in calculating matter hamiltonian

}

} // namespace nuTens