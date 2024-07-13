
#include <nuTens/tensors/tensor.hpp>


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



Tensor &Tensor::dType(NTdtypes::scalarType type){
    _tensor = _tensor.to(scalarTypeMap[type]);
    return *this;
}

Tensor &Tensor::device(NTdtypes::deviceType device){
    _tensor = _tensor.to(deviceTypeMap[device]);
    return *this;
}

Tensor &Tensor::requiresGrad(bool reqGrad){
    _tensor = _tensor.set_requires_grad(reqGrad);
    return *this;
}


Tensor Tensor::getValue(const std::vector<std::variant<int, std::string>> &indices) const {
    std::vector<at::indexing::TensorIndex> indicesVec;
    for(size_t i = 0; i < indices.size(); i++){
        if (const int* index = std::get_if<int>(&indices[i]))
            indicesVec.push_back(at::indexing::TensorIndex(*index));
        else if (const std::string *index = std::get_if<std::string>(&indices[i]))
            indicesVec.push_back(at::indexing::TensorIndex((*index).c_str()));
        else{
            assert(false && "ERROR: Unsupported index type");
            throw;
        }
    }

    Tensor ret;
    ret._tensor = _tensor.index(indicesVec);
    return ret;
}

void Tensor::setValue(const Tensor &indices, const Tensor &value){
    _tensor.index_put_({indices._tensor}, value._tensor);
}

void Tensor::setValue(const std::vector<std::variant<int, std::string>> &indices, const Tensor &value){
    std::vector<at::indexing::TensorIndex> indicesVec;
    for(size_t i = 0; i < indices.size(); i++){
        if (const int* index = std::get_if<int>(&indices[i]))
            indicesVec.push_back(at::indexing::TensorIndex(*index));
        else if (const std::string *index = std::get_if<std::string>(&indices[i]))
            indicesVec.push_back(at::indexing::TensorIndex((*index).c_str()));
        else{
            assert(false && "ERROR: Unsupported index type");
            throw;
        }
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

size_t Tensor::getNdim() const {
    return _tensor._dimI();
}

int Tensor::getBatchDim() const {
    return _tensor.sizes()[0];
}

std::vector<int> Tensor::getShape() const{
    std::vector<int> ret(getNdim());
    for ( size_t i = 0; i < getNdim(); i++){
        ret[i] = _tensor.sizes()[i];
    }
    return ret;
}

Tensor Tensor::matmul(const Tensor &t1, const Tensor &t2){
    Tensor ret;
    ret._tensor = torch::matmul(t1._tensor, t2._tensor);
    return ret;
}

Tensor Tensor::outer(const Tensor &t1, const Tensor &t2){
    Tensor ret;
    ret._tensor = torch::outer(t1._tensor, t2._tensor);
    return ret;
}

Tensor Tensor::mul(const Tensor &t1, const Tensor &t2){
    Tensor ret;
    ret._tensor = torch::mul(t1._tensor, t2._tensor);
    return ret;
}

Tensor Tensor::div(const Tensor &t1, const Tensor &t2){
    Tensor ret;
    ret._tensor = torch::div(t1._tensor, t2._tensor);
    return ret;
}

Tensor Tensor::pow(const Tensor &t, float s){
    Tensor ret;
    ret._tensor = torch::pow(t._tensor, s);
    return ret;
}

Tensor Tensor::pow(const Tensor &t, std::complex<float> s){
    Tensor ret;
    ret._tensor = torch::pow(t._tensor, c10::complex<float>(s.real(), s.imag()));
    return ret;
}

Tensor Tensor::exp(const Tensor &t){
    Tensor ret;
    ret._tensor = torch::exp(t._tensor);
    return ret;
}

Tensor Tensor::transpose(const Tensor &t, int dim1, int dim2){
    Tensor ret;
    ret._tensor = torch::transpose(t._tensor, dim1, dim2);
    return ret;
}

Tensor Tensor::scale(const Tensor &t, float s){
    Tensor ret;
    ret._tensor = torch::multiply(t._tensor, s);
    return ret;
}

Tensor Tensor::scale(const Tensor &t, std::complex<float> s){
    Tensor ret;
    ret._tensor = torch::multiply(t._tensor, c10::complex<float>(s.real(), s.imag()));
    return ret;
}


void Tensor::matmul_(const Tensor &t2){
    _tensor = torch::matmul(_tensor, t2._tensor);
}

void Tensor::mul_(const Tensor &t2){
    _tensor = torch::mul(_tensor, t2._tensor);
}

void Tensor::div_(const Tensor &t2){
    _tensor = torch::div(_tensor, t2._tensor);
}

void Tensor::scale_(float s){
    _tensor = torch::multiply(_tensor, s);
}

void Tensor::scale_(std::complex<float> s){
    _tensor = torch::multiply(_tensor, c10::complex<float>(s.real(), s.imag()));
}


void Tensor::pow_(float s){
    _tensor = torch::pow(_tensor, s);
}

void Tensor::pow_(std::complex<float> s){
    _tensor = torch::pow(_tensor, c10::complex<float>(s.real(), s.imag()));
}

void Tensor::exp_(){
    _tensor = torch::exp(_tensor);
}

void Tensor::transpose_(int dim1, int dim2){
    _tensor = torch::transpose(_tensor, dim1, dim2);
}


void Tensor::eig(const Tensor &t, Tensor &eVals, Tensor &eVecs){

    auto ret = torch::linalg_eig(t._tensor);
    eVals._tensor = std::get<1>(ret);
    eVecs._tensor = std::get<0>(ret);
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

Tensor Tensor::conj() const {
    Tensor ret;
    ret._tensor = torch::conj(_tensor);
    // torch::conj() returns a view of the original tensor
    // I *think* that means that the tensor returned here will be pointing to the same memory as the original one
    // might need to be careful with this  
    return ret;
}

Tensor Tensor::abs() const {
    Tensor ret;
    ret._tensor = torch::abs(_tensor);
    return ret;
}

Tensor Tensor::angle() const {
    Tensor ret;
    ret._tensor = torch::angle(_tensor);
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


Tensor Tensor::cumsum(int dim) const {
    Tensor ret;
    ret._tensor = torch::cumsum(_tensor, dim);
    return ret;
}

Tensor Tensor::sum() const {
    Tensor ret;
    ret._tensor = _tensor.sum();
    return ret;
}

void Tensor::backward() const {
    _tensor.backward();
}

Tensor Tensor::grad() const {
    Tensor ret;
    ret._tensor = _tensor.grad();
    return ret;
}


Tensor Tensor::sin(const Tensor &t) {
    Tensor ret;
    ret._tensor = torch::sin(t._tensor);
    return ret;
}

Tensor Tensor::cos(const Tensor &t) {
    Tensor ret;
    ret._tensor = torch::cos(t._tensor);
    return ret;
}

std::string Tensor::toString() const {
    std::ostringstream stream;
    stream << _tensor;
    return stream.str();
}