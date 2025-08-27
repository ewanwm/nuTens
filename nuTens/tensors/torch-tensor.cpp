
#include <nuTens/tensors/tensor.hpp>

using namespace nuTens;

std::string Tensor::getTensorLibrary()
{
    return "PyTorch";
}

Tensor::Tensor(const std::vector<float> &values, dtypes::scalarType type, dtypes::deviceType device, bool requiresGrad)
    : _dType(type), _device(device)
{
    NT_PROFILE();

    _tensor = torch::tensor(values, torch::TensorOptions()
                                        .dtype(dtypes::scalarTypeMap(type))
                                        .device(dtypes::deviceTypeMap(device))
                                        .requires_grad(requiresGrad));
}

Tensor Tensor::eye(int n, dtypes::scalarType type, dtypes::deviceType device, bool requiresGrad)
{
    NT_PROFILE();

    Tensor ret;
    ret.setTensor(torch::eye(n, torch::TensorOptions()
                                    .dtype(dtypes::scalarTypeMap(type))
                                    .device(dtypes::deviceTypeMap(device))
                                    .requires_grad(requiresGrad)));
    ret._dType = type;
    ret._device = device;
    return ret;
}

Tensor Tensor::rand(const std::vector<long int> &shape, dtypes::scalarType type, dtypes::deviceType device,
                    bool requiresGrad)
{
    NT_PROFILE();

    Tensor ret;
    ret.setTensor(torch::rand(c10::IntArrayRef(shape), torch::TensorOptions()
                                                           .dtype(dtypes::scalarTypeMap(type))
                                                           .device(dtypes::deviceTypeMap(device))
                                                           .requires_grad(requiresGrad)));

    ret._dType = type;
    ret._device = device;
    return ret;
}

Tensor Tensor::diag(const Tensor &diag)
{
    assert(diag.getNdim() == 1);
    NT_PROFILE();

    Tensor ret;
    ret.setTensor(torch::diag(diag._tensor));
    ret._dType = diag._dType;
    ret._device = diag._device;
    return ret;
}

Tensor Tensor::ones(const std::vector<long int> &shape, dtypes::scalarType type, dtypes::deviceType device,
                    bool requiresGrad)
{
    NT_PROFILE();

    Tensor ret;
    ret.setTensor(torch::ones(c10::IntArrayRef(shape), torch::TensorOptions()
                                                           .dtype(dtypes::scalarTypeMap(type))
                                                           .device(dtypes::deviceTypeMap(device))
                                                           .requires_grad(requiresGrad)));
    ret._dType = type;
    ret._device = device;
    return ret;
}

Tensor Tensor::zeros(const std::vector<long int> &shape, dtypes::scalarType type, dtypes::deviceType device,
                     bool requiresGrad)
{
    NT_PROFILE();

    Tensor ret;
    ret.setTensor(torch::zeros(c10::IntArrayRef(shape), torch::TensorOptions()
                                                            .dtype(dtypes::scalarTypeMap(type))
                                                            .device(dtypes::deviceTypeMap(device))
                                                            .requires_grad(requiresGrad)));
    ret._dType = type;
    ret._device = device;
    return ret;
}

Tensor &Tensor::dType(dtypes::scalarType type)
{
    NT_PROFILE();

    _tensor = _tensor.to(dtypes::scalarTypeMap(type));
    _dType = type;
    return *this;
}

Tensor &Tensor::device(dtypes::deviceType device)
{
    NT_PROFILE();

    _tensor = _tensor.to(dtypes::deviceTypeMap(device));
    _device = device;
    return *this;
}

Tensor &Tensor::requiresGrad(bool reqGrad)
{
    NT_PROFILE();

    _tensor = _tensor.set_requires_grad(reqGrad);
    return *this;
}

Tensor &Tensor::addBatchDim()
{
    NT_PROFILE();

    if (!_hasBatchDim)
    {
        _tensor = torch::unsqueeze(_tensor, 0);
        _hasBatchDim = true;
    }

    return *this;
}

Tensor &Tensor::unsqueeze(int index)
{
    NT_PROFILE();

    _tensor = torch::unsqueeze(_tensor, index);

    return *this;
}

Tensor Tensor::getValues(const std::vector<Tensor::indexType> &indices) const
{
    NT_PROFILE();

    Tensor ret;
    ret.setTensor(_tensor.index(convertIndices(indices)));
    return ret;
}

Tensor::variantType Tensor::getVariantValue(const std::vector<int> &indices) const
{
    NT_PROFILE();

    switch (_dType)
    {
    case dtypes::kInt:
        return _tensor.index(convertIndices(indices)).item<int>();

    case dtypes::kFloat:
        return _tensor.index(convertIndices(indices)).item<float>();

    case dtypes::kDouble:
        return _tensor.index(convertIndices(indices)).item<double>();

    case dtypes::kComplexFloat:
        return (std::complex<float>)_tensor.index(convertIndices(indices)).item<c10::complex<float>>();

    case dtypes::kComplexDouble:
        return (std::complex<double>)_tensor.index(convertIndices(indices)).item<c10::complex<double>>();

    default:
        NT_ERROR("Invalid dtype has been set for this tensor: {}", _dType);
        NT_ERROR("{}:{}", __FILE__, __LINE__);
        throw;
    }
}

void Tensor::setValue(const Tensor &indices, const Tensor &value)
{
    NT_PROFILE();

    _tensor.index_put_({indices._tensor}, value._tensor);
}

void Tensor::setValue(const std::vector<Tensor::indexType> &indices, const Tensor &value)
{
    NT_PROFILE();

    _tensor.index_put_(convertIndices(indices), value._tensor);
}

void Tensor::setValue(const std::vector<int> &indices, float value)
{
    NT_PROFILE();

    _tensor.index_put_(convertIndices(indices), value);
}

void Tensor::setValue(const std::vector<int> &indices, double value)
{
    NT_PROFILE();

    _tensor.index_put_(convertIndices(indices), value);
}

void Tensor::setValue(const std::vector<int> &indices, std::complex<float> value)
{
    NT_PROFILE();

    _tensor.index_put_(convertIndices(indices), c10::complex<float>(value.real(), value.imag()));
}

void Tensor::setValue(const std::vector<int> &indices, std::complex<double> value)
{
    NT_PROFILE();

    _tensor.index_put_(convertIndices(indices), c10::complex<double>(value.real(), value.imag()));
}

size_t Tensor::getNdim() const
{
    NT_PROFILE();

    return _tensor.dim();
}

int Tensor::getBatchDim() const
{
    NT_PROFILE();

    return _tensor.sizes()[0];
}

bool Tensor::getHasBatchDim() const
{
    NT_PROFILE();

    return _hasBatchDim;
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
    ret.setTensor(torch::matmul(t1._tensor, t2._tensor));
    return ret;
}

Tensor Tensor::outer(const Tensor &t1, const Tensor &t2)
{
    NT_PROFILE();

    Tensor ret;
    ret.setTensor(torch::outer(t1._tensor, t2._tensor));
    return ret;
}

Tensor Tensor::mul(const Tensor &t1, const Tensor &t2)
{
    NT_PROFILE();

    Tensor ret;
    ret.setTensor(torch::mul(t1._tensor, t2._tensor));
    return ret;
}

Tensor Tensor::div(const Tensor &t1, const Tensor &t2)
{
    NT_PROFILE();

    Tensor ret;
    ret.setTensor(torch::div(t1._tensor, t2._tensor));
    return ret;
}

Tensor Tensor::pow(const Tensor &t, float s)
{
    NT_PROFILE();

    Tensor ret;
    ret.setTensor(torch::pow(t._tensor, s));
    return ret;
}

Tensor Tensor::pow(const Tensor &t, std::complex<float> s)
{
    NT_PROFILE();

    Tensor ret;
    ret.setTensor(torch::pow(t._tensor, c10::complex<float>(s.real(), s.imag())));
    return ret;
}

Tensor Tensor::exp(const Tensor &t)
{
    NT_PROFILE();

    Tensor ret;
    ret.setTensor(torch::exp(t._tensor));
    return ret;
}

Tensor Tensor::transpose(const Tensor &t, int dim1, int dim2)
{
    NT_PROFILE();

    Tensor ret;
    ret.setTensor(torch::transpose(t._tensor, dim1, dim2));
    return ret;
}

Tensor Tensor::scale(const Tensor &t, float s)
{
    NT_PROFILE();

    Tensor ret;
    ret.setTensor(torch::multiply(t._tensor, s));
    return ret;
}

Tensor Tensor::scale(const Tensor &t, double s)
{
    NT_PROFILE();

    Tensor ret;
    ret.setTensor(torch::multiply(t._tensor, s));
    return ret;
}

Tensor Tensor::scale(const Tensor &t, std::complex<float> s)
{
    NT_PROFILE();

    assert(t._dType == dtypes::kComplexFloat | t._dType == dtypes::kComplexDouble);

    Tensor ret;
    ret.setTensor(torch::multiply(t._tensor, c10::complex<float>(s.real(), s.imag())));
    return ret;
}

Tensor Tensor::scale(const Tensor &t, std::complex<double> s)
{
    NT_PROFILE();

    assert(t._dType == dtypes::kComplexFloat | t._dType == dtypes::kComplexDouble);

    Tensor ret;
    ret.setTensor(torch::multiply(t._tensor, c10::complex<double>(s.real(), s.imag())));
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
    eVals.setTensor(std::get<0>(ret));
    eVecs.setTensor(std::get<1>(ret));
}

void Tensor::eigh(const Tensor &t, Tensor &eVals, Tensor &eVecs)
{
    NT_PROFILE();

    auto ret = torch::linalg_eigh(t._tensor);
    eVals.setTensor(std::get<0>(ret));
    eVecs.setTensor(std::get<1>(ret));
}

void Tensor::eigvals(const Tensor &t, Tensor &eVals)
{
    NT_PROFILE();

    eVals.setTensor(torch::linalg_eigvals(t._tensor));
}

void Tensor::eigvalsh(const Tensor &t, Tensor &eVals)
{
    NT_PROFILE();

    eVals.setTensor(torch::linalg_eigvalsh(t._tensor));
}

void Tensor::qr(const Tensor &t, Tensor &Q, Tensor &R)
{
    NT_PROFILE();

    auto ret = torch::linalg::qr(t._tensor);
    Q.setTensor(std::get<1>(ret));
    R.setTensor(std::get<0>(ret));
}

Tensor Tensor::real() const
{
    NT_PROFILE();

    Tensor ret;
    ret.setTensor(at::real(_tensor));
    return ret;
}

Tensor Tensor::imag() const
{
    NT_PROFILE();

    Tensor ret;
    ret.setTensor(at::imag(_tensor));
    return ret;
}

Tensor Tensor::conj() const
{
    NT_PROFILE();

    Tensor ret;
    ret.setTensor(torch::conj(_tensor));
    // torch::conj() returns a view of the original tensor
    // I *think* that means that the tensor returned here will be pointing to the
    // same memory as the original one might need to be careful with this
    return ret;
}

Tensor Tensor::abs() const
{
    NT_PROFILE();

    Tensor ret;
    ret.setTensor(torch::abs(_tensor));
    return ret;
}

Tensor Tensor::angle() const
{
    NT_PROFILE();

    Tensor ret;
    ret.setTensor(torch::angle(_tensor));
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
    ret.setTensor(_tensor + rhs._tensor);
    return ret;
}

Tensor Tensor::operator+(double rhs) const
{
    NT_PROFILE();

    Tensor ret;
    ret.setTensor(_tensor + rhs);
    return ret;
}

Tensor Tensor::operator-(const Tensor &rhs) const
{
    NT_PROFILE();

    Tensor ret;
    ret.setTensor(_tensor - rhs._tensor);
    return ret;
}

Tensor Tensor::operator-(double rhs) const
{
    NT_PROFILE();

    Tensor ret;
    ret.setTensor(_tensor - rhs);
    return ret;
}

Tensor Tensor::operator-() const
{
    NT_PROFILE();

    Tensor ret;
    ret.setTensor(-_tensor);
    return ret;
}

Tensor Tensor::operator*(const Tensor &rhs) const
{
    NT_PROFILE();

    Tensor ret;
    ret.setTensor(_tensor * rhs._tensor);
    return ret;
}

Tensor Tensor::operator*(double rhs) const
{
    NT_PROFILE();

    Tensor ret;
    ret.setTensor(_tensor * rhs);
    return ret;
}

Tensor Tensor::operator/(const Tensor &rhs) const
{
    NT_PROFILE();

    Tensor ret;
    ret.setTensor(_tensor / rhs._tensor);
    return ret;
}

Tensor Tensor::operator/(double rhs) const
{
    NT_PROFILE();

    Tensor ret;
    ret.setTensor(_tensor / rhs);
    return ret;
}

Tensor Tensor::cumsum(int dim) const
{
    NT_PROFILE();

    Tensor ret;
    ret.setTensor(torch::cumsum(_tensor, dim));
    return ret;
}

Tensor Tensor::sum() const
{
    NT_PROFILE();

    Tensor ret;
    ret.setTensor(_tensor.sum());
    return ret;
}

Tensor Tensor::sum(const std::vector<long int> &dims) const
{
    NT_PROFILE();

    Tensor ret;
    ret.setTensor(torch::sum(_tensor, torch::OptionalArrayRef<long int>(dims)));
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
    ret.setTensor(_tensor.grad());
    return ret;
}

Tensor Tensor::sin(const Tensor &t)
{
    NT_PROFILE();

    Tensor ret;
    ret.setTensor(torch::sin(t._tensor));
    return ret;
}

Tensor Tensor::cos(const Tensor &t)
{
    NT_PROFILE();

    Tensor ret;
    ret.setTensor(torch::cos(t._tensor));
    return ret;
}

std::string Tensor::toString() const
{
    NT_PROFILE();

    std::ostringstream stream;
    stream << _tensor;
    return stream.str();
}