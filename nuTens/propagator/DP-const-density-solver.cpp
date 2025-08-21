#include <nuTens/propagator/DP-const-density-solver.hpp>

using namespace nuTens;

void DPconstDensitySolver::calculateIntermediate() {

	// Ueisq's
	Ue2sq = Tensor::mul(cosSqTheta13, sinSqTheta12);
	Ue3sq = sinSqTheta13;

	// Umisq's, Utisq's and Jvac	 
	Um3sq = Tensor::mul(cosSqTheta13, sinSqTheta23);

	// Um2sq and Ut2sq are used here as temporary variables, will be properly defined later	 
	Ut2sq = Tensor::mul(Tensor::mul(sinSqTheta13, sinSqTheta12), sinSqTheta23);

	Jrr = Tensor::pow( cosSqTheta12 * cosSqTheta23 * Ut2sq, 0.5);

	Um2sq = cosSqTheta12 * cosSqTheta23 + Ut2sq - Jrr * cosDeltaCP * 2.0;
	Jmatter = Jrr * cosSqTheta13 * sinDeltaCP * 8.0;
	Amatter = _energies * _density * constants::Groot2 * 2.0;
	Dmsqee = _dmsq31 - sinSqTheta12 * _dmsq21;

	// calculate A, B, C, See, Tee, and part of Tmm
	Araw = _dmsq21 + _dmsq31; // temporary variable
	A = Araw + Amatter;
	See = Araw - _dmsq21 * Ue2sq - _dmsq31 * Ue3sq;
	Tmm = _dmsq21 * _dmsq31; // using Tmm as a temporary variable	  
	Tee = Tmm * (one -  Ue3sq - Ue2sq);
	C = Amatter * Tee;
}

void DPconstDensitySolver::calculateEigenvalues(Tensor &lambda1, Tensor &lambda2, Tensor &lambda3, Tensor &Dlambda21, Tensor &Dlambda31, Tensor &Dlambda32) {

	// ---------------------------------- //
	// Get lambda3 from lambda+ of MP/DMP //
	// ---------------------------------- //
	Tensor xmat = Amatter / Dmsqee;
	Tensor tmp = one - xmat;
	lambda3 = _dmsq31 + Dmsqee * (xmat - 1 + Tensor::pow(tmp * tmp + sinSqTheta13 * xmat * 4.0, 0.5)) * 0.5;

	// ---------------------------------------------------------------------------- //
	// Newton iterations to improve lambda3 arbitrarily, if needed, (B needed here) //
	// ---------------------------------------------------------------------------- //
	Tensor B = Tmm + Amatter * See; // B is only needed for N_Newton >= 1
	for (int i = 0; i < _NRiterations; i++)
		lambda3 = (lambda3 * lambda3 * (lambda3 + lambda3 - A) + C) / (lambda3 * ((lambda3 - A) * 2.0 + lambda3) + B); // this strange form prefers additions to multiplications
	
	// ------------------- //
	// Get  Delta lambda's //
	// ------------------- //
	Dlambda21 = Tensor::pow( (A - lambda3) * (A - lambda3) - C * 4.0 / lambda3, 0.5);
	lambda1 = (A - lambda3 - Dlambda21) * 0.5;
	lambda2 = (A - lambda3 + Dlambda21) * 0.5;
	Dlambda32 = lambda3 - lambda2;
	Dlambda31 = Dlambda32 + Dlambda21;
}

Tensor DPconstDensitySolver::calculateProbs()
{
	NT_PROFILE();

	Tensor lambda1, lambda2, lambda3, Dlambda21, Dlambda31, Dlambda32;
	calculateEigenvalues(lambda1, lambda2, lambda3, Dlambda21, Dlambda31, Dlambda32);

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

	Tensor Smm = A - _dmsq21 * Um2sq - _dmsq31 * Um3sq;
	Tmm = Tmm * (one - Um3sq - Um2sq) + Amatter * (See + Smm - A);

	Um3sq = (lambda3 * (lambda3 - Smm) + Tmm) * Xp3;
	Um2sq = (lambda2 * (lambda2 - Smm) + Tmm) * Xp2;

	// ------------- //
	// Use NHS for J //
	// ------------- //
	Jmatter = Jmatter * _dmsq21 * _dmsq31 * (_dmsq31 - _dmsq21) * PiDlambdaInv;

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

	Tensor D21 = Dlambda21 * _baseline / (_energies * 4.0);
	Tensor D32 = Dlambda32 * _baseline / (_energies * 4.0);
	  
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
	Tensor Pme_CPC = (Ut3sq - Um2sq * Ue1sq - Um1sq * Ue2sq) * sinsqD21_2
			       + (Ut2sq - Um3sq * Ue1sq - Um1sq * Ue3sq) * sinsqD31_2
			       + (Ut1sq - Um3sq * Ue2sq - Um2sq * Ue3sq) * sinsqD32_2;
	Tensor Pme_CPV = -Jmatter * triple_sin;

	Tensor Pmm = one - (Um2sq * Um1sq * sinsqD21_2
				      + Um3sq * Um1sq * sinsqD31_2
				      + Um3sq * Um2sq * sinsqD32_2) * 2.0;

	Tensor Pee = one - (Ue2sq * Ue1sq * sinsqD21_2
				      + Ue3sq * Ue1sq * sinsqD31_2
				      + Ue3sq * Ue2sq * sinsqD32_2) * 2.0;


	Tensor probsRet = Tensor::zeros({_energies.getShape()[0], 3, 3}).requiresGrad(false);

	// ---------------------------- //
	// Assign all the probabilities //
	// ---------------------------- //
	probsRet.setValue({"...", 0, 0}, Pee.getValues({"...", 0}));                                                        // Pee
	probsRet.setValue({"...", 0, 1}, (Pme_CPC - Pme_CPV).getValues({"...", 0}));                                        // Pem
	probsRet.setValue({"...", 0, 2}, (one - Pee - Pme_CPC + Pme_CPV).getValues({"...", 0}));                            // Pet

	probsRet.setValue({"...", 1, 0}, (Pme_CPC + Pme_CPV).getValues({"...", 0}));                                        // Pme
	probsRet.setValue({"...", 1, 1}, Pmm.getValues({"...", 0}));                                                        // Pmm
	probsRet.setValue({"...", 1, 2}, (one - Pme_CPC - Pme_CPV - Pmm).getValues({"...", 0}));                            // Pmt

	probsRet.setValue({"...", 2, 0}, (one - Pee - Pme_CPC - Pme_CPV).getValues({"...", 0}));                            // Pte
	probsRet.setValue({"...", 2, 1}, (one - Pme_CPC + Pme_CPV - Pmm).getValues({"...", 0}));                            // Ptm
	probsRet.setValue({"...", 2, 2}, (one - (one - Pee - Pme_CPC + Pme_CPV) - (one - Pme_CPC - Pme_CPV - Pmm)).getValues({"...", 0}));  // Ptt
	
	return probsRet;
}
