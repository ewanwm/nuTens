import torch
import math as m
import os

GROOT2 = 0.76294e-4 
TWOPI = 2.0 * m.pi

class DPpropagator(torch.nn.Module):

    def __init__(self, antiNeutrino, nr_iterations, density, baseline):

        super().__init__()
        
        self._antiNeutrino = antiNeutrino
        self._nr_iterations = nr_iterations
        self._density = density
        self._baseline = baseline

    def forward(self, sinSqTheta12, sinSqTheta13, sinSqTheta23, dmsq31, dmsq21, deltaCP, energies):

        antinuFactor = (0.5 - (float(self._antiNeutrino))) * 2.0

        # --------------------------------------------------------------------- //
        # First calculate useful simple functions of the oscillation parameters //
        # --------------------------------------------------------------------- //

        cosSqTheta12 = 1.0 - sinSqTheta12
        cosSqTheta13 = 1.0 - sinSqTheta13
        cosSqTheta23 = 1.0 - sinSqTheta23

        sinDeltaCP = torch.sin(deltaCP)
        cosDeltaCP = torch.cos(deltaCP)

        # Ueisq's
        Ue2sq = cosSqTheta13 * sinSqTheta12
        Ue3sq = sinSqTheta13

        # if user wants to interpret theta_ij's as sin^2(theta_ij) we use the "normal" nufast method
        # for Jrr, effectively forcing the thetas to be in lower octant.
        # Otherwise we calculate performing the trig functions which is slower but allows any octant
        Jrr = None

        Jrr = torch.sqrt(cosSqTheta12 * cosSqTheta23 * sinSqTheta13 * sinSqTheta12 * sinSqTheta23)
        

        # Umisq's, Utisq's and Jvac
        Um2sq = cosSqTheta12 * cosSqTheta23 + sinSqTheta13 * sinSqTheta12 * sinSqTheta23 - Jrr * cosDeltaCP * 2.0
        Um3sq = cosSqTheta13 * sinSqTheta23

        Amatter = energies * (antinuFactor * self._density * GROOT2 * 2.0)
        Dmsqee = -dmsq31 + sinSqTheta12 * dmsq21

        # calculate Atotal, Bmatter, Cmatter, See, Tee, and part of Tmm
        Araw = -dmsq21 - dmsq31
        Atotal = Araw + Amatter

        See = Araw + dmsq21 * Ue2sq + dmsq31 * Ue3sq
        Tee = dmsq21 * dmsq31 * (1.0 - Ue3sq - Ue2sq)

        Smm = Atotal + dmsq21 * Um2sq + dmsq31 * Um3sq
        Tmm = dmsq21 * dmsq31 * (1.0 - Um3sq - Um2sq) + Amatter * (See + Smm - Atotal)
        
        Cmatter = Amatter * Tee

        # ---------------------------------- //
        # Get lambda3 from lambda+ of MP/DMP //
        # ---------------------------------- //
        xmat = Amatter / Dmsqee
        tmp = 1.0 - xmat
        # NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)
        lambda3 = -dmsq31 + Dmsqee * (xmat - 1 + torch.sqrt(tmp * tmp + sinSqTheta13 * xmat * 4.0)) * 0.5
        # NOLINTEND(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

        # ---------------------------------------------------------------------------- //
        # Newton iterations to improve lambda3 arbitrarily, if needed, (Bmatter needed here) //
        # ---------------------------------------------------------------------------- //
        Bmatter = dmsq21 * dmsq31 + Amatter * See # Bmatter is only needed for N_Newton >= 1
        for i in range(self._nr_iterations):
            lambda3 = (lambda3 * lambda3 * (lambda3 + lambda3 - Atotal) + Cmatter) / (lambda3 * ((lambda3 - Atotal) * 2.0 + lambda3) + Bmatter) # this strange form prefers additions to multiplications

        # ------------------- //
        # Get  Delta lambda's //
        # ------------------- //
        tmp = Atotal - lambda3
        Dlambda21 = torch.sqrt(tmp * tmp - Cmatter * 4.0 / lambda3)
        lambda2 = (Atotal - lambda3 + Dlambda21) * 0.5
        Dlambda32 = lambda3 - lambda2
        Dlambda31 = Dlambda32 + Dlambda21

        # ----------------------- //
        # Use Rosetta for Veisq's //
        # ----------------------- //
        # denominators
        PiDlambdaInv = 1.0 / (Dlambda31 * Dlambda32 * Dlambda21)
        Xp3 = PiDlambdaInv * Dlambda21
        Xp2 = -PiDlambdaInv * Dlambda31

        # numerators
        Ve3sq = (lambda3 * (lambda3 - See) + Tee) * Xp3
        Ve2sq = (lambda2 * (lambda2 - See) + Tee) * Xp2

        Vm3sq = (lambda3 * (lambda3 - Smm) + Tmm) * Xp3
        Vm2sq = (lambda2 * (lambda2 - Smm) + Tmm) * Xp2

        # ----------------------- //
        # Get all elements of Usq //
        # ----------------------- //
        Ve1sq = 1.0 - Ve3sq - Ve2sq
        Vm1sq = 1.0 - Vm3sq - Vm2sq

        Vt3sq = 1.0 - Vm3sq - Ve3sq
        Vt2sq = 1.0 - Vm2sq - Ve2sq
        Vt1sq = 1.0 - Vm1sq - Ve1sq

        # ----------------------- //
        # Get the kinematic terms //
        # ----------------------- //

        D21 = Dlambda21 * self._baseline * TWOPI / (energies * antinuFactor * 4.0)
        D32 = Dlambda32 * self._baseline * TWOPI / (energies * antinuFactor * 4.0)

        sinD21 = torch.sin(D21)
        sinD31 = torch.sin(D32 + D21)
        sinD32 = torch.sin(D32)

        triple_sin = sinD21 * sinD31 * sinD32

        sinsqD21_2 = sinD21 * sinD21 * 2.0
        sinsqD31_2 = sinD31 * sinD31 * 2.0
        sinsqD32_2 = sinD32 * sinD32 * 2.0

        # ------------- //
        # Use NHS for J //
        # ------------- //
        Jmatter = Jrr * cosSqTheta13 * sinDeltaCP * 8.0 * dmsq21 * dmsq31 * (dmsq21 - dmsq31) * PiDlambdaInv

        # ------------------------------------------------------------------- //
        # Calculate the three necessary probabilities, separating CPC and CPV //
        # ------------------------------------------------------------------- //
        Pme_CPC = (Vt3sq - Vm2sq * Ve1sq - Vm1sq * Ve2sq) * sinsqD21_2 + (Vt2sq - Vm3sq * Ve1sq - Vm1sq * Ve3sq) * sinsqD31_2 + (Vt1sq - Vm3sq * Ve2sq - Vm2sq * Ve3sq) * sinsqD32_2
        Pme_CPV = -Jmatter * triple_sin

        Pmm = 1.0 - (Vm2sq * Vm1sq * sinsqD21_2 + Vm3sq * Vm1sq * sinsqD31_2 + Vm3sq * Vm2sq * sinsqD32_2) * 2.0

        Pee = 1.0 - (Ve2sq * Ve1sq * sinsqD21_2 + Ve3sq * Ve1sq * sinsqD31_2 + Ve3sq * Ve2sq * sinsqD32_2) * 2.0

        # ---------------------------- //
        # Assign all the probabilities //
        # ---------------------------- //

        probs_ret = torch.stack(
            [
                torch.stack(
                    [
                        Pee,                             # Pee
                        (Pme_CPC - Pme_CPV),             # Pem
                        (1.0 - Pee - Pme_CPC + Pme_CPV)  # Pet
                    ],
                    dim = -1
                ),
                torch.stack(
                    [
                        (Pme_CPC + Pme_CPV),             # Pme
                        Pmm,                             # Pmm
                        (1.0 - Pme_CPC - Pme_CPV - Pmm)  # Pmt
                    ],
                    dim = -1
                ),
                torch.stack(
                    [
                        (1.0 - Pee - Pme_CPC - Pme_CPV), # Pte
                        (1.0 - Pme_CPC + Pme_CPV - Pmm), # Ptm
                        (Pee + 2.0 * Pme_CPC - 1.0 + Pmm) # Ptt
                    ],
                    dim = -1
                )
            ],
            dim = 1
        )

        return probs_ret.contiguous()

with torch.no_grad():
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")

    for device in devices:
        model = DPpropagator(False, 10, 2.6, 295_000).to(device=device).to(torch.float32)
        example_inputs=(
            torch.rand(1, device=device, dtype = torch.float32),
            torch.rand(1, device=device, dtype = torch.float32),
            torch.rand(1, device=device, dtype = torch.float32),
            torch.rand(1, device=device, dtype = torch.float32),
            torch.rand(1, device=device, dtype = torch.float32),
            torch.rand(1, device=device, dtype = torch.float32),
            torch.rand(100, device=device, dtype = torch.float32),
        )

        print(model(*example_inputs))
        batch_dim = torch.export.Dim("batch", min=1, max=1024)
        # [Optional] Specify the first dimension of the input x as dynamic.
        exported = torch.export.export(
            model, 
            example_inputs, 
            dynamic_shapes={
                "sinSqTheta12": {}, #{0: batch_dim}, 
                "sinSqTheta13": {}, #{0: batch_dim}, 
                "sinSqTheta23": {}, #{0: batch_dim},
                "dmsq31":       {}, #{0: batch_dim}, 
                "dmsq21":       {}, #{0: batch_dim}, 
                "deltaCP":      {}, #{0: batch_dim},
                "energies":     {0: batch_dim},
            }
        )
        # [Note] In this example we directly feed the exported module to aoti_compile_and_package.
        # Depending on your use case, e.g. if your training platform and inference platform
        # are different, you may choose to save the exported model using torch.export.save and
        # then load it back using torch.export.load on your inference platform to run AOT compilation.
        output_path = torch._inductor.aoti_compile_and_package(
            exported,
            # [Optional] Specify the generated shared library path. If not specified,
            # the generated artifact is stored in your system temp directory.
            package_path=os.path.join(os.getcwd(), f"precompiledDPpropagator-{device}.pt2"),
            # [Optional] Specify Inductor configs
            # This specific max_autotune option will turn on more extensive kernel autotuning for
            # better performance.
            inductor_configs={"max_autotune": True,},
        )
