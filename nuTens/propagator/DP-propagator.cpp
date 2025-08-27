#include <nuTens/propagator/DP-propagator.hpp>

using namespace nuTens;

Tensor DPpropagator::calculateProbs()
{
    NT_PROFILE();

    float antinuFactor = (0.5 - ((float)_antiNeutrino)) * 2.0;

    // --------------------------------------------------------------------- //
    // First calculate useful simple functions of the oscillation parameters //
    // --------------------------------------------------------------------- //
    Tensor one = Tensor::ones({1}).requiresGrad(false);

    Tensor sinSqTheta12 = Tensor::pow(Tensor::sin(theta12), 2.0);
    Tensor cosSqTheta12 = Tensor::pow(Tensor::cos(theta12), 2.0);
    Tensor sinSqTheta13 = Tensor::pow(Tensor::sin(theta13), 2.0);
    Tensor cosSqTheta13 = Tensor::pow(Tensor::cos(theta13), 2.0);
    Tensor sinSqTheta23 = Tensor::pow(Tensor::sin(theta23), 2.0);
    Tensor cosSqTheta23 = Tensor::pow(Tensor::cos(theta23), 2.0);

    Tensor sinDeltaCP = Tensor::sin(deltaCP);
    Tensor cosDeltaCP = Tensor::cos(deltaCP);

    // Ueisq's
    Tensor Ue2sq = Tensor::mul(cosSqTheta13, sinSqTheta12);
    Tensor Ue3sq = sinSqTheta13;

    // Umisq's, Utisq's and Jvac
    Tensor Um3sq = Tensor::mul(cosSqTheta13, sinSqTheta23);

    // Um2sq and Ut2sq are used here as temporary variables, will be properly defined later
    Tensor Ut2sq = Tensor::mul(Tensor::mul(sinSqTheta13, sinSqTheta12), sinSqTheta23);
    Tensor Um2sq = Tensor::mul(cosSqTheta12, cosSqTheta23);

    /// TODO: The nufast version of this would look like
    ///         Tensor::pow( Tensor::mul(Um2sq, Ut2sq), 0.5);
    ///       however this means that the sign of the sin functions is lost
    ///       giving weong Um2sq and then wrong osc probs
    Tensor Jrr = Tensor::cos(theta12) * Tensor::cos(theta23) * Tensor::sin(theta13) * Tensor::sin(theta12) *
                 Tensor::sin(theta23);

    Um2sq = Um2sq + Ut2sq - Jrr * cosDeltaCP * 2.0;
    Tensor Jmatter = Jrr * cosSqTheta13 * sinDeltaCP * 8.0;
    Tensor Amatter = _energies * antinuFactor * _density * constants::Groot2 * 2.0;
    Tensor Dmsqee = -dmsq31 + sinSqTheta12 * dmsq21;

    // calculate A, B, C, See, Tee, and part of Tmm
    Tensor A = -dmsq21 - dmsq31; // temporary variable
    Tensor See = A + dmsq21 * Ue2sq + dmsq31 * Ue3sq;
    Tensor Tmm = dmsq21 * dmsq31; // using Tmm as a temporary variable
    Tensor Tee = Tmm * (one - Ue3sq - Ue2sq);
    Tensor C = Amatter * Tee;
    A = A + Amatter;

    // ---------------------------------- //
    // Get lambda3 from lambda+ of MP/DMP //
    // ---------------------------------- //
    Tensor xmat = Amatter / Dmsqee;
    Tensor tmp = one - xmat;
    Tensor lambda3 = -dmsq31 + Dmsqee * (xmat - 1 + Tensor::pow(tmp * tmp + sinSqTheta13 * xmat * 4.0, 0.5)) * 0.5;

    // ---------------------------------------------------------------------------- //
    // Newton iterations to improve lambda3 arbitrarily, if needed, (B needed here) //
    // ---------------------------------------------------------------------------- //
    Tensor B = Tmm + Amatter * See; // B is only needed for N_Newton >= 1
    for (int i = 0; i < NRiterations; i++)
        lambda3 =
            (lambda3 * lambda3 * (lambda3 + lambda3 - A) + C) /
            (lambda3 * ((lambda3 - A) * 2.0 + lambda3) + B); // this strange form prefers additions to multiplications

    // ------------------- //
    // Get  Delta lambda's //
    // ------------------- //
    tmp = A - lambda3;
    Tensor Dlambda21 = Tensor::pow(tmp * tmp - C * 4.0 / lambda3, 0.5);
    Tensor lambda2 = (A - lambda3 + Dlambda21) * 0.5;
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
    Ue3sq = (lambda3 * (lambda3 - See) + Tee) * Xp3;
    Ue2sq = (lambda2 * (lambda2 - See) + Tee) * Xp2;

    Tensor Smm = A + dmsq21 * Um2sq + dmsq31 * Um3sq;
    Tmm = Tmm * (one - Um3sq - Um2sq) + Amatter * (See + Smm - A);

    Um3sq = (lambda3 * (lambda3 - Smm) + Tmm) * Xp3;
    Um2sq = (lambda2 * (lambda2 - Smm) + Tmm) * Xp2;

    // ------------- //
    // Use NHS for J //
    // ------------- //
    Jmatter = Jmatter * dmsq21 * dmsq31 * (dmsq21 - dmsq31) * PiDlambdaInv;

    // ----------------------- //
    // Get all elements of Usq //
    // ----------------------- //
    Tensor Ue1sq = one - Ue3sq - Ue2sq;
    Tensor Um1sq = one - Um3sq - Um2sq;

    Tensor Ut3sq = one - Um3sq - Ue3sq;
    Ut2sq = one - Um2sq - Ue2sq;
    Tensor Ut1sq = one - Um1sq - Ue1sq;

    // ----------------------- //
    // Get the kinematic terms //
    // ----------------------- //

    Tensor D21 = Dlambda21 * _baseline * 2.0 * M_PI / (_energies * antinuFactor * 4.0);
    Tensor D32 = Dlambda32 * _baseline * 2.0 * M_PI / (_energies * antinuFactor * 4.0);

    Tensor sinD21 = Tensor::sin(D21);
    Tensor sinD31 = Tensor::sin(D32 + D21);
    Tensor sinD32 = Tensor::sin(D32);

    Tensor triple_sin = sinD21 * sinD31 * sinD32;

    Tensor sinsqD21_2 = sinD21 * sinD21 * 2.0;
    Tensor sinsqD31_2 = sinD31 * sinD31 * 2.0;
    Tensor sinsqD32_2 = sinD32 * sinD32 * 2.0;

    // ------------------------------------------------------------------- //
    // Calculate the three necessary probabilities, separating CPC and CPV //
    // ------------------------------------------------------------------- //
    Tensor Pme_CPC = (Ut3sq - Um2sq * Ue1sq - Um1sq * Ue2sq) * sinsqD21_2 +
                     (Ut2sq - Um3sq * Ue1sq - Um1sq * Ue3sq) * sinsqD31_2 +
                     (Ut1sq - Um3sq * Ue2sq - Um2sq * Ue3sq) * sinsqD32_2;
    Tensor Pme_CPV = -Jmatter * triple_sin;

    Tensor Pmm = one - (Um2sq * Um1sq * sinsqD21_2 + Um3sq * Um1sq * sinsqD31_2 + Um3sq * Um2sq * sinsqD32_2) * 2.0;

    Tensor Pee = one - (Ue2sq * Ue1sq * sinsqD21_2 + Ue3sq * Ue1sq * sinsqD31_2 + Ue3sq * Ue2sq * sinsqD32_2) * 2.0;

    Tensor probsRet = Tensor::zeros({_energies.getShape()[0], 3, 3}).requiresGrad(false);

    // ---------------------------- //
    // Assign all the probabilities //
    // ---------------------------- //
    probsRet.setValue({"...", 0, 0}, Pee.getValues({"...", 0}));                             // Pee
    probsRet.setValue({"...", 0, 1}, (Pme_CPC - Pme_CPV).getValues({"...", 0}));             // Pem
    probsRet.setValue({"...", 0, 2}, (one - Pee - Pme_CPC + Pme_CPV).getValues({"...", 0})); // Pet

    probsRet.setValue({"...", 1, 0}, (Pme_CPC + Pme_CPV).getValues({"...", 0}));             // Pme
    probsRet.setValue({"...", 1, 1}, Pmm.getValues({"...", 0}));                             // Pmm
    probsRet.setValue({"...", 1, 2}, (one - Pme_CPC - Pme_CPV - Pmm).getValues({"...", 0})); // Pmt

    probsRet.setValue({"...", 2, 0}, (one - Pee - Pme_CPC - Pme_CPV).getValues({"...", 0})); // Pte
    probsRet.setValue({"...", 2, 1}, (one - Pme_CPC + Pme_CPV - Pmm).getValues({"...", 0})); // Ptm
    probsRet.setValue(
        {"...", 2, 2},
        (one - (one - Pee - Pme_CPC + Pme_CPV) - (one - Pme_CPC - Pme_CPV - Pmm)).getValues({"...", 0})); // Ptt

    return probsRet;
}