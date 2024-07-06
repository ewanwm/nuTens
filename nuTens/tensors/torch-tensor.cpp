
#include <nuTens/tensors/tensor.hpp>
#include <torch/torch.h>


// map between the data types used in nuTens and those used by pytorch
std::map<NTdtypes::scalarType, c10::ScalarType> scalarTypeMap = {
    {NTdtypes::kFloat, torch::kFloat},
    {NTdtypes::kDouble, torch::kDouble},
    {NTdtypes::kComplexFloat, torch::kComplexFloat},
    {NTdtypes::kComplexDouble, torch::kComplexDouble}
};

// map between the device types used in nuTens and those used by pytorch
std::map<NTdtypes::deviceType, c10::DeviceType> deviceTypeMap = {
    {NTdtypes::kCPU, torch::kCPU},
    {NTdtypes::kGPU, torch::kCUDA}
};


std::string Tensor::getTensorLibrary(){
    return "PyTorch";
}

Tensor &Tensor::ones(int length, NTdtypes::scalarType type, NTdtypes::deviceType device, bool requiresGrad){
    _tensor = 
        torch::ones(
            length, 
            torch::TensorOptions().dtype(scalarTypeMap[type]).device(deviceTypeMap[device]).requires_grad(requiresGrad)
        );

    return *this;
}

Tensor &Tensor::ones(const std::vector<long int> &shape, NTdtypes::scalarType type, NTdtypes::deviceType device, bool requiresGrad){
    _tensor = 
        torch::ones(
            c10::IntArrayRef(shape), 
            torch::TensorOptions().dtype(scalarTypeMap[type]).device(deviceTypeMap[device]).requires_grad(requiresGrad)
        );
    return *this;
}

Tensor &Tensor::zeros(int length, NTdtypes::scalarType type, NTdtypes::deviceType device, bool requiresGrad){
    _tensor = 
        torch::zeros(length, scalarTypeMap[type]);
    return *this;
}

Tensor &Tensor::zeros(const std::vector<long int> &shape, NTdtypes::scalarType type, NTdtypes::deviceType device, bool requiresGrad){
    _tensor = 
        torch::zeros(c10::IntArrayRef(shape), scalarTypeMap[type]);
    return *this;
}



Tensor& Tensor::dType(NTdtypes::scalarType type){
    _tensor = _tensor.to(scalarTypeMap[type]);
    return *this;
}

Tensor& Tensor::device(NTdtypes::deviceType device){
    _tensor = _tensor.to(deviceTypeMap[device]);
    return *this;
}


void Tensor::setValue(const Tensor &indices, const Tensor &value){
    _tensor.index_put_({indices._tensor}, value._tensor);
}

void Tensor::setValue(const std::vector<int> &indices, const Tensor &value){
    std::vector<at::indexing::TensorIndex> indicesVec;
    for(size_t i = 0; i < indices.size(); i++){
        indicesVec.push_back(at::indexing::TensorIndex(indices[i]));
    }
    
    _tensor.index_put_(indicesVec, value._tensor);
}

void Tensor::setValue(const std::vector<int> &indices, float value){
    std::vector<at::indexing::TensorIndex> indicesVec;
    for(size_t i = 0; i < indices.size(); i++){
        indicesVec.push_back(at::indexing::TensorIndex(indices[i]));
    }

    _tensor.index_put_(indicesVec, value);
}

void Tensor::setValue(const std::vector<int> &indices, std::complex<float> value){
    std::vector<at::indexing::TensorIndex> indicesVec;
    for(size_t i = 0; i < indices.size(); i++){
        indicesVec.push_back(at::indexing::TensorIndex(indices[i]));
    }

    _tensor.index_put_(indicesVec, c10::complex<float>(value.real(), value.imag()));
}

Tensor Tensor::matmul(const Tensor &t1, const Tensor &t2){
    Tensor ret;
    ret._tensor = torch::matmul(t1._tensor, t2._tensor);
    return ret;
}

void Tensor::matmul_(const Tensor &t2){
    _tensor = torch::matmul(_tensor, t2._tensor);
}

Tensor Tensor::real()const {
    Tensor ret;
    ret._tensor = at::real(_tensor);
    return ret;
}

Tensor Tensor::imag() const {
    Tensor ret;
    ret._tensor = at::imag(_tensor);
    return ret;
}

bool Tensor::operator== (const Tensor &rhs) const {
    return at::equal(_tensor, rhs._tensor);
}

bool Tensor::operator!= (const Tensor &rhs) const {
    return !at::equal(_tensor, rhs._tensor);
}

Tensor Tensor::operator+ (const Tensor &rhs) const {
    Tensor ret;
    ret._tensor = _tensor + rhs._tensor;
    return ret;
}

Tensor Tensor::operator- (const Tensor &rhs) const {
    Tensor ret;
    ret._tensor = _tensor - rhs._tensor;
    return ret;
}

Tensor Tensor::operator- () const {
    Tensor ret;
    ret._tensor = -_tensor;
    return ret;
}

std::string Tensor::toString() const {
    std::ostringstream stream;
    stream << _tensor;
    return stream.str();
}