#include <nuTens/propagator/DP-propagator.hpp>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include <torch/torch.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>

using namespace nuTens;

class PrecompiledDPpropagator : public DPpropagator
{

    void loadModelCPU()
    {

        extern char _binary_precompiledDPpropagator_cpu_pt2_start;
        extern char _binary_precompiledDPpropagator_cpu_pt2_end;
        extern int _binary_precompiledDPpropagator_cpu_pt2_size;

        int size = (size_t)&_binary_precompiledDPpropagator_cpu_pt2_size;

        const char *start = (const char *)&_binary_precompiledDPpropagator_cpu_pt2_start;
        const char *end = (const char *)&_binary_precompiledDPpropagator_cpu_pt2_end;

        std::cout << size << " vs " << end - start << std::endl;

        std::string tempName = std::tmpnam(nullptr);

        std::cout << "temp file: " << tempName << std::endl;
        std::ofstream tempfile(tempName, std::ios::out | std::ios::binary);

        tempfile.write(start, size);

        propagator = std::make_unique<torch::inductor::AOTIModelPackageLoader>(tempName);

        // delete the temp file
        std::remove(tempName.c_str());
    }

    std::unique_ptr<torch::inductor::AOTIModelPackageLoader> propagator;    //{"DPpropagator-cpu.pt2"};
    std::unique_ptr<torch::inductor::AOTIModelPackageLoader> propagatorGPU; //{"DPpropagator-cuda.pt2"};

  public:
    PrecompiledDPpropagator(int NRiterations, dtypes::deviceType device = dtypes::kCPU)
        : DPpropagator(NRiterations, device)
    {
        loadModelCPU();
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
            outputs = propagatorGPU->run(inputs);
        }
        else
        {
            outputs = propagator->run(inputs);
        }

        Tensor ret = Tensor::fromTorchTensor(outputs[0].contiguous());

        return ret;
    }
};
