#include <nuTens/propagator/DP-propagator.hpp>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include <torch/torch.h>

using namespace nuTens;

class PrecompiledDPpropagator : public DPpropagator
{

    torch::inductor::AOTIModelPackageLoader propagator{"DPpropagator-cpu.pt2"};
    torch::inductor::AOTIModelPackageLoader propagatorGPU{"DPpropagator-cuda.pt2"};

  public:
    PrecompiledDPpropagator(int NRiterations, dtypes::deviceType device = dtypes::kCPU)
        : DPpropagator(NRiterations, device)
    {
    }

    [[nodiscard]] virtual inline Tensor calculateProbs() override
    {
        NT_PROFILE();

        std::vector<torch::Tensor> inputs = {
            theta12.getTensor().contiguous(),   theta13.getTensor().contiguous(), theta23.getTensor().contiguous(),
            dmsq31.getTensor().contiguous(),    dmsq21.getTensor().contiguous(),  deltaCP.getTensor().contiguous(),
            _energies.getTensor().contiguous(),
        };

        std::cout << _energies.getTensor() << std::endl;
        std::vector<torch::Tensor> outputs;

        if (_device == dtypes::kGPU)
        {
            outputs = propagatorGPU.run(inputs);
        }
        else
        {
            outputs = propagator.run(inputs);
        }

        Tensor ret = Tensor::fromTorchTensor(outputs[0].contiguous());

        return ret;
    }
};
