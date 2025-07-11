#pragma once

/// @file constants.hpp
/// @brief Defines constants to be used across the project

#include <nuTens/propagator/units.hpp>

namespace Constants
{

    static constexpr float Groot2 = 1.52588e-4 * (Units::eV * Units::eV) / Units::GeV; //!< sqrt(2)*G_fermi in (eV^2-cm^3)/(mole-GeV) used in calculating matter hamiltonian

}