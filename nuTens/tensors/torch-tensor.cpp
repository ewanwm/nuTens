
#include <nuTens/tensors/tensor.hpp>

// map between the data types used in nuTens and those used by pytorch
const static std::map<NTdtypes::scalarType, c10::ScalarType> scalarTypeMap = {
    {NTdtypes::kFloat, torch::kFloat},
    {NTdtypes::kDouble, torch::kDouble},
    {NTdtypes::kComplexFloat, torch::kComplexFloat},
    {NTdtypes::kComplexDouble, torch::kComplexDouble}};

// map between the device types used in nuTens and those used by pytorch
const static std::map<NTdtypes::deviceType, c10::DeviceType> deviceTypeMap = {{NTdtypes::kCPU, torch::kCPU},
                                                                              {NTdtypes::kGPU, torch::kCUDA}};

std::string Tensor::getTensorLibrary()
{
    return "PyTorch";
}

Tensor &Tensor::ones(int length, NTdtypes::scalarType type, NTdtypes::deviceType device, bool requiresGrad)
{
    NT_PROFILE();

    _tensor = torch::ones(length, torch::TensorOptions()
                                      .dtype(scalarTypeMap.at(type))
                                      .device(deviceTypeMap.at(device))
                                      .requires_grad(requiresGrad));

    return *this;
}

Tensor &Tensor::ones(const std::vector<long int> &shape, NTdtypes::scalarType type, NTdtypes::deviceType device,
                     bool requiresGrad)
{
    NT_PROFILE();

    _tensor = torch::ones(c10::IntArrayRef(shape), torch::TensorOptions()
                                                       .dtype(scalarTypeMap.at(type))
                                                       .device(deviceTypeMap.at(device))
                                                       .requires_grad(requiresGrad));
    return *this;
}

Tensor &Tensor::zeros(int length, NTdtypes::scalarType type, NTdtypes::deviceType device, bool requiresGrad)
{
    NT_PROFILE();

    _tensor = torch::zeros(length, scalarTypeMap.at(type));
    return *this;
}

Tensor &Tensor::zeros(const std::vector<long int> &shape, NTdtypes::scalarType type, NTdtypes::deviceType device,
                      bool requiresGrad)
{
    NT_PROFILE();

    _tensor = torch::zeros(c10::IntArrayRef(shape), scalarTypeMap.at(type));
    return *this;
}

Tensor &Tensor::dType(NTdtypes::scalarType type)
{
    NT_PROFILE();

    _tensor = _tensor.to(scalarTypeMap.at(type));
    return *this;
}

Tensor &Tensor::device(NTdtypes::deviceType device)
{
    NT_PROFILE();

    _tensor = _tensor.to(deviceTypeMap.at(device));
    return *this;
}

Tensor &Tensor::requiresGrad(bool reqGrad)
{
    NT_PROFILE();

    _tensor = _tensor.set_requires_grad(reqGrad);
    return *this;
}

Tensor Tensor::getValue(const std::vector<Tensor::indexType> &indices) const
{
    NT_PROFILE();

    std::vector<at::indexing::TensorIndex> indicesVec;
    for (const Tensor::indexType &i : indices)
    {
        if (const int *index = std::get_if<int>(&i))
        {
            indicesVec.push_back(at::indexing::TensorIndex(*index));
        }
        else if (const std::string *index = std::get_if<std::string>(&i))
        {
            indicesVec.push_back(at::indexing::TensorIndex((*index).c_str()));
        }
        else
        {
            assert(false && "ERROR: Unsupported index type");
            throw;
        }
    }

    Tensor ret;
    ret._tensor = _tensor.index(indicesVec);
    return ret;
}

void Tensor::setValue(const Tensor &indices, const Tensor &value)
{
    NT_PROFILE();

    _tensor.index_put_({indices._tensor}, value._tensor);
}

void Tensor::setValue(const std::vector<Tensor::indexType> &indices, const Tensor &value)
{
    NT_PROFILE();

    std::vector<at::indexing::TensorIndex> indicesVec;
    for (const Tensor::indexType &i : indices)
    {
        if (const int *index = std::get_if<int>(&i))
        {
            indicesVec.push_back(at::indexing::TensorIndex(*index));
        }
        else if (const std::string *index = std::get_if<std::string>(&i))
        {
            indicesVec.push_back(at::indexing::TensorIndex((*index).c_str()));
        }
        else
        {
            assert(false && "ERROR: Unsupported index type");
            throw;
        }
    }

    _tensor.index_put_(indicesVec, value._tensor);
}

void Tensor::setValue(const std::vector<int> &indices, float value)
{
    NT_PROFILE();

    std::vector<at::indexing::TensorIndex> indicesVec;
    indicesVec.reserve(indices.size());
    for (const int &i : indices)
    {
        indicesVec.push_back(at::indexing::TensorIndex(i));
    }

    _tensor.index_put_(indicesVec, value);
}

void Tensor::setValue(const std::vector<int> &indices, std::complex<float> value)
{
    NT_PROFILE();

    std::vector<at::indexing::TensorIndex> indicesVec;
    indicesVec.reserve(indices.size());
    for (const int &i : indices)
    {
        indicesVec.push_back(at::indexing::TensorIndex(i));
    }

    _tensor.index_put_(indicesVec, c10::complex<float>(value.real(), value.imag()));
}

size_t Tensor::getNdim() const
{
    NT_PROFILE();

    return _tensor._dimI();
}

int Tensor::getBatchDim() const
{
    NT_PROFILE();

    return _tensor.sizes()[0];
}

std::vector<int> Tensor::getShape() const
{
    NT_PROFILE();

    std::vector<int> ret(getNdim());
    for (size_t i = 0; i < getNdim(); i++)
    {
        ret[i] = _tensor.sizes()[i];
    }
    return ret;
}

Tensor Tensor::matmul(const Tensor &t1, const Tensor &t2)
{
    NT_PROFILE();

    Tensor ret;
    ret._tensor = torch::matmul(t1._tensor, t2._tensor);
    return ret;
}

Tensor Tensor::outer(const Tensor &t1, const Tensor &t2)
{
    NT_PROFILE();

    Tensor ret;
    ret._tensor = torch::outer(t1._tensor, t2._tensor);
    return ret;
}

Tensor Tensor::mul(const Tensor &t1, const Tensor &t2)
{
    NT_PROFILE();

    Tensor ret;
    ret._tensor = torch::mul(t1._tensor, t2._tensor);
    return ret;
}

Tensor Tensor::div(const Tensor &t1, const Tensor &t2)
{
    NT_PROFILE();

    Tensor ret;
    ret._tensor = torch::div(t1._tensor, t2._tensor);
    return ret;
}

Tensor Tensor::pow(const Tensor &t, float s)
{
    NT_PROFILE();

    Tensor ret;
    ret._tensor = torch::pow(t._tensor, s);
    return ret;
}

Tensor Tensor::pow(const Tensor &t, std::complex<float> s)
{
    NT_PROFILE();

    Tensor ret;
    ret._tensor = torch::pow(t._tensor, c10::complex<float>(s.real(), s.imag()));
    return ret;
}

Tensor Tensor::exp(const Tensor &t)
{
    NT_PROFILE();

    Tensor ret;
    ret._tensor = torch::exp(t._tensor);
    return ret;
}

Tensor Tensor::transpose(const Tensor &t, int dim1, int dim2)
{
    NT_PROFILE();

    Tensor ret;
    ret._tensor = torch::transpose(t._tensor, dim1, dim2);
    return ret;
}

Tensor Tensor::scale(const Tensor &t, float s)
{
    NT_PROFILE();

    Tensor ret;
    ret._tensor = torch::multiply(t._tensor, s);
    return ret;
}

Tensor Tensor::scale(const Tensor &t, std::complex<float> s)
{
    NT_PROFILE();

    Tensor ret;
    ret._tensor = torch::multiply(t._tensor, c10::complex<float>(s.real(), s.imag()));
    return ret;
}

void Tensor::matmul_(const Tensor &t2)
{
    NT_PROFILE();

    _tensor = torch::matmul(_tensor, t2._tensor);
}

void Tensor::mul_(const Tensor &t2)
{
    NT_PROFILE();

    _tensor = torch::mul(_tensor, t2._tensor);
}

void Tensor::div_(const Tensor &t2)
{
    NT_PROFILE();

    _tensor = torch::div(_tensor, t2._tensor);
}

void Tensor::scale_(float s)
{
    NT_PROFILE();

    _tensor = torch::multiply(_tensor, s);
}

void Tensor::scale_(std::complex<float> s)
{
    NT_PROFILE();

    _tensor = torch::multiply(_tensor, c10::complex<float>(s.real(), s.imag()));
}

void Tensor::pow_(float s)
{
    NT_PROFILE();

    _tensor = torch::pow(_tensor, s);
}

void Tensor::pow_(std::complex<float> s)
{
    NT_PROFILE();

    _tensor = torch::pow(_tensor, c10::complex<float>(s.real(), s.imag()));
}

void Tensor::exp_()
{
    NT_PROFILE();

    _tensor = torch::exp(_tensor);
}

void Tensor::transpose_(int dim1, int dim2)
{
    NT_PROFILE();

    _tensor = torch::transpose(_tensor, dim1, dim2);
}

void Tensor::eig(const Tensor &t, Tensor &eVals, Tensor &eVecs)
{
    NT_PROFILE();

    auto ret = torch::linalg_eig(t._tensor);
    eVals._tensor = std::get<1>(ret);
    eVecs._tensor = std::get<0>(ret);
}

Tensor Tensor::real() const
{
    NT_PROFILE();

    Tensor ret;
    ret._tensor = at::real(_tensor);
    return ret;
}

Tensor Tensor::imag() const
{
    NT_PROFILE();

    Tensor ret;
    ret._tensor = at::imag(_tensor);
    return ret;
}

Tensor Tensor::conj() const
{
    NT_PROFILE();

    Tensor ret;
    ret._tensor = torch::conj(_tensor);
    // torch::conj() returns a view of the original tensor
    // I *think* that means that the tensor returned here will be pointing to the
    // same memory as the original one might need to be careful with this
    return ret;
}

Tensor Tensor::abs() const
{
    NT_PROFILE();

    Tensor ret;
    ret._tensor = torch::abs(_tensor);
    return ret;
}

Tensor Tensor::angle() const
{
    NT_PROFILE();

    Tensor ret;
    ret._tensor = torch::angle(_tensor);
    return ret;
}

bool Tensor::operator==(const Tensor &rhs) const
{
    NT_PROFILE();

    return at::equal(_tensor, rhs._tensor);
}

bool Tensor::operator!=(const Tensor &rhs) const
{
    NT_PROFILE();

    return !at::equal(_tensor, rhs._tensor);
}

Tensor Tensor::operator+(const Tensor &rhs) const
{
    NT_PROFILE();

    Tensor ret;
    ret._tensor = _tensor + rhs._tensor;
    return ret;
}

Tensor Tensor::operator-(const Tensor &rhs) const
{
    NT_PROFILE();

    Tensor ret;
    ret._tensor = _tensor - rhs._tensor;
    return ret;
}

Tensor Tensor::operator-() const
{
    NT_PROFILE();

    Tensor ret;
    ret._tensor = -_tensor;
    return ret;
}

Tensor Tensor::cumsum(int dim) const
{
    NT_PROFILE();

    Tensor ret;
    ret._tensor = torch::cumsum(_tensor, dim);
    return ret;
}

Tensor Tensor::sum() const
{
    NT_PROFILE();

    Tensor ret;
    ret._tensor = _tensor.sum();
    return ret;
}

void Tensor::backward() const
{
    NT_PROFILE();

    _tensor.backward();
}

Tensor Tensor::grad() const
{
    NT_PROFILE();

    Tensor ret;
    ret._tensor = _tensor.grad();
    return ret;
}

Tensor Tensor::sin(const Tensor &t)
{
    NT_PROFILE();

    Tensor ret;
    ret._tensor = torch::sin(t._tensor);
    return ret;
}

Tensor Tensor::cos(const Tensor &t)
{
    NT_PROFILE();

    Tensor ret;
    ret._tensor = torch::cos(t._tensor);
    return ret;
}

std::string Tensor::toString() const
{
    NT_PROFILE();

    std::ostringstream stream;
    stream << _tensor;
    return stream.str();
}