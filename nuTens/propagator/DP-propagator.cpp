#include <nuTens/propagator/DP-propagator.hpp>

using namespace nuTens;

Tensor DPpropagator::calculateProbs()
{
    NT_PROFILE();

    float antinuFactor = (0.5 - ((float)_antiNeutrino)) * 2.0;

    // --------------------------------------------------------------------- //
    // First calculate useful simple functions of the oscillation parameters //
    // --------------------------------------------------------------------- //
    const Tensor one = Tensor::ones({1}).requiresGrad(false);

    // need to calculate the sin^2(theta)'s if not provided by user
    if (!interpretSinSquaredThetas)
    {
        sinSqTheta12 = Tensor::pow(Tensor::sin(theta12), 2.0);
        sinSqTheta13 = Tensor::pow(Tensor::sin(theta13), 2.0);
        sinSqTheta23 = Tensor::pow(Tensor::sin(theta23), 2.0);
    }

    Tensor cosSqTheta12 = one - sinSqTheta12;
    Tensor cosSqTheta13 = one - sinSqTheta13;
    Tensor cosSqTheta23 = one - sinSqTheta23;

    Tensor sinDeltaCP = Tensor::sin(deltaCP);
    Tensor cosDeltaCP = Tensor::cos(deltaCP);

    // Ueisq's
    Tensor Ue2sq = cosSqTheta13 * sinSqTheta12;
    Tensor Ue3sq = sinSqTheta13;

    // if user wants to interpret theta_ij's as sin^2(theta_ij) we use the "normal" nufast method
    // for Jrr, effectively forcing the thetas to be in lower octant.
    // Otherwise we calculate performing the trig functions which is slower but allows any octant
    Tensor Jrr;

    if (interpretSinSquaredThetas)
    {
        Jrr = Tensor::sqrt(cosSqTheta12 * cosSqTheta23 * sinSqTheta13 * sinSqTheta12 * sinSqTheta23);
    }
    else
    {
        Jrr = Tensor::cos(theta12) * Tensor::cos(theta23) * Tensor::sin(theta13) * Tensor::sin(theta12) *
              Tensor::sin(theta23);
    }

    // Umisq's, Utisq's and Jvac
    Tensor Um2sq = cosSqTheta12 * cosSqTheta23 + sinSqTheta13 * sinSqTheta12 * sinSqTheta23 - Jrr * cosDeltaCP * 2.0;
    Tensor Um3sq = cosSqTheta13 * sinSqTheta23;

    Tensor Amatter = _energies * (antinuFactor * _density * constants::Groot2 * 2.0);
    Tensor Dmsqee = -dmsq31 + sinSqTheta12 * dmsq21;

    // calculate Atotal, Bmatter, Cmatter, See, Tee, and part of Tmm
    Tensor Araw = -dmsq21 - dmsq31;
    Tensor Atotal = Araw + Amatter;

    Tensor See = Araw + dmsq21 * Ue2sq + dmsq31 * Ue3sq;
    Tensor Tee = dmsq21 * dmsq31 * (one - Ue3sq - Ue2sq);

    Tensor Smm = Atotal + dmsq21 * Um2sq + dmsq31 * Um3sq;
    Tensor Tmm = dmsq21 * dmsq31 * (one - Um3sq - Um2sq) + Amatter * (See + Smm - Atotal);

    Tensor Cmatter = Amatter * Tee;

    // ---------------------------------- //
    // Get lambda3 from lambda+ of MP/DMP //
    // ---------------------------------- //
    Tensor xmat = Amatter / Dmsqee;
    Tensor tmp = one - xmat;
    // NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)
    Tensor lambda3{-dmsq31 + Dmsqee * (xmat - 1 + Tensor::sqrt(tmp * tmp + sinSqTheta13 * xmat * 4.0)) * 0.5};
    // NOLINTEND(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

    // ---------------------------------------------------------------------------- //
    // Newton iterations to improve lambda3 arbitrarily, if needed, (Bmatter needed here) //
    // ---------------------------------------------------------------------------- //
    Tensor Bmatter = dmsq21 * dmsq31 + Amatter * See; // Bmatter is only needed for N_Newton >= 1
    for (int i = 0; i < NRiterations; i++)
    {
        lambda3 = (lambda3 * lambda3 * (lambda3 + lambda3 - Atotal) + Cmatter) /
                  (lambda3 * ((lambda3 - Atotal) * 2.0 + lambda3) +
                   Bmatter); // this strange form prefers additions to multiplications
    }

    // ------------------- //
    // Get  Delta lambda's //
    // ------------------- //
    tmp = Atotal - lambda3;
    Tensor Dlambda21 = Tensor::sqrt(tmp * tmp - Cmatter * 4.0 / lambda3);
    Tensor lambda2 = (Atotal - lambda3 + Dlambda21) * 0.5;
    Tensor Dlambda32 = lambda3 - lambda2;
    Tensor Dlambda31 = Dlambda32 + Dlambda21;

    // ----------------------- //
    // Use Rosetta for Veisq's //
    // ----------------------- //
    // denominators
    Tensor PiDlambdaInv = one / (Dlambda31 * Dlambda32 * Dlambda21);
    Tensor Xp3 = PiDlambdaInv * Dlambda21;
    Tensor Xp2 = -PiDlambdaInv * Dlambda31;

    // numerators
    Tensor Ve3sq = (lambda3 * (lambda3 - See) + Tee) * Xp3;
    Tensor Ve2sq = (lambda2 * (lambda2 - See) + Tee) * Xp2;

    Tensor Vm3sq = (lambda3 * (lambda3 - Smm) + Tmm) * Xp3;
    Tensor Vm2sq = (lambda2 * (lambda2 - Smm) + Tmm) * Xp2;

    // ----------------------- //
    // Get all elements of Usq //
    // ----------------------- //
    Tensor Ve1sq = one - Ve3sq - Ve2sq;
    Tensor Vm1sq = one - Vm3sq - Vm2sq;

    Tensor Vt3sq = one - Vm3sq - Ve3sq;
    Tensor Vt2sq = one - Vm2sq - Ve2sq;
    Tensor Vt1sq = one - Vm1sq - Ve1sq;

    // ----------------------- //
    // Get the kinematic terms //
    // ----------------------- //

    Tensor D21 = Dlambda21 * _baseline * constants::twoPi / (_energies * antinuFactor * 4.0);
    Tensor D32 = Dlambda32 * _baseline * constants::twoPi / (_energies * antinuFactor * 4.0);

    Tensor sinD21 = Tensor::sin(D21);
    Tensor sinD31 = Tensor::sin(D32 + D21);
    Tensor sinD32 = Tensor::sin(D32);

    Tensor triple_sin = sinD21 * sinD31 * sinD32;

    Tensor sinsqD21_2 = sinD21 * sinD21 * 2.0;
    Tensor sinsqD31_2 = sinD31 * sinD31 * 2.0;
    Tensor sinsqD32_2 = sinD32 * sinD32 * 2.0;

    // ------------- //
    // Use NHS for J //
    // ------------- //
    Tensor Jmatter = Jrr * cosSqTheta13 * sinDeltaCP * 8.0 * dmsq21 * dmsq31 * (dmsq21 - dmsq31) * PiDlambdaInv;

    // ------------------------------------------------------------------- //
    // Calculate the three necessary probabilities, separating CPC and CPV //
    // ------------------------------------------------------------------- //
    Tensor Pme_CPC = (Vt3sq - Vm2sq * Ve1sq - Vm1sq * Ve2sq) * sinsqD21_2 +
                     (Vt2sq - Vm3sq * Ve1sq - Vm1sq * Ve3sq) * sinsqD31_2 +
                     (Vt1sq - Vm3sq * Ve2sq - Vm2sq * Ve3sq) * sinsqD32_2;
    Tensor Pme_CPV = -Jmatter * triple_sin;

    Tensor Pmm = one - (Vm2sq * Vm1sq * sinsqD21_2 + Vm3sq * Vm1sq * sinsqD31_2 + Vm3sq * Vm2sq * sinsqD32_2) * 2.0;

    Tensor Pee = one - (Ve2sq * Ve1sq * sinsqD21_2 + Ve3sq * Ve1sq * sinsqD31_2 + Ve3sq * Ve2sq * sinsqD32_2) * 2.0;

    Tensor probsRet = Tensor::zeros({_energies.getShape()[0], 3, 3}).requiresGrad(false);

    // ---------------------------- //
    // Assign all the probabilities //
    // ---------------------------- //
    probsRet.setValue({"...", 0, 0}, Pee);                             // Pee
    probsRet.setValue({"...", 0, 1}, (Pme_CPC - Pme_CPV));             // Pem
    probsRet.setValue({"...", 0, 2}, (one - Pee - Pme_CPC + Pme_CPV)); // Pet

    probsRet.setValue({"...", 1, 0}, (Pme_CPC + Pme_CPV));             // Pme
    probsRet.setValue({"...", 1, 1}, Pmm);                             // Pmm
    probsRet.setValue({"...", 1, 2}, (one - Pme_CPC - Pme_CPV - Pmm)); // Pmt

    probsRet.setValue({"...", 2, 0}, (one - Pee - Pme_CPC - Pme_CPV)); // Pte
    probsRet.setValue({"...", 2, 1}, (one - Pme_CPC + Pme_CPV - Pmm)); // Ptm
    probsRet.setValue({"...", 2, 2},
                      (Pee + 2.0 * Pme_CPC - one + Pmm)); // Ptt

    return probsRet;
}