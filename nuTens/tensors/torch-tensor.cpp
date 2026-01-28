
#include <nuTens/tensors/tensor.hpp>

using namespace nuTens;

// LCOV_EXCL_START
std::string Tensor::getTensorLibrary()
{
    return "PyTorch";
}
// LCOV_EXCL_STOP

Tensor::Tensor(const std::vector<float> &values, dtypes::scalarType type, dtypes::deviceType device, bool requiresGrad)
    : _dType(type), _device(device), _requiresGrad(requiresGrad), _initialised(true)
{
    NT_PROFILE();

    _tensor = torch::tensor(values, torch::TensorOptions()
                                        .dtype(dtypes::scalarTypeMap(type))
                                        .device(dtypes::deviceTypeMap(device))
                                        .requires_grad(requiresGrad));
}

Tensor Tensor::TensorComplex(const std::vector<std::complex<float>> &values, dtypes::scalarType type,
                             dtypes::deviceType device, bool requiresGrad)
{
    NT_PROFILE();

    std::vector<c10::complex<float>> c10Values;
    c10Values.reserve(values.size());
    for (const auto &value : values)
    {
        c10Values.push_back(c10::complex<float>(value.real(), value.imag()));
    }

    return {torch::tensor(c10Values, torch::TensorOptions()
                                         .dtype(dtypes::scalarTypeMap(type))
                                         .device(dtypes::deviceTypeMap(device))
                                         .requires_grad(requiresGrad))};
}

Tensor Tensor::eye(int n, dtypes::scalarType type, dtypes::deviceType device, bool requiresGrad)
{
    NT_PROFILE();

    return {torch::eye(n, torch::TensorOptions()
                              .dtype(dtypes::scalarTypeMap(type))
                              .device(dtypes::deviceTypeMap(device))
                              .requires_grad(requiresGrad))};
}

Tensor Tensor::rand(const std::vector<long int> &shape, dtypes::scalarType type, dtypes::deviceType device,
                    bool requiresGrad)
{
    NT_PROFILE();

    return {torch::rand(c10::IntArrayRef(shape), torch::TensorOptions()
                                                     .dtype(dtypes::scalarTypeMap(type))
                                                     .device(dtypes::deviceTypeMap(device))
                                                     .requires_grad(requiresGrad))};
}

Tensor Tensor::diag(const Tensor &diag)
{
    assert(diag.getNdim() == 1);
    NT_PROFILE();

    return {torch::diag(diag._tensor)};
}

Tensor Tensor::ones(const std::vector<long int> &shape, dtypes::scalarType type, dtypes::deviceType device,
                    bool requiresGrad)
{
    NT_PROFILE();

    return {torch::ones(c10::IntArrayRef(shape), torch::TensorOptions()
                                                     .dtype(dtypes::scalarTypeMap(type))
                                                     .device(dtypes::deviceTypeMap(device))
                                                     .requires_grad(requiresGrad))};
}

Tensor Tensor::zeros(const std::vector<long int> &shape, dtypes::scalarType type, dtypes::deviceType device,
                     bool requiresGrad)
{
    NT_PROFILE();

    return {torch::zeros(c10::IntArrayRef(shape), torch::TensorOptions()
                                                      .dtype(dtypes::scalarTypeMap(type))
                                                      .device(dtypes::deviceTypeMap(device))
                                                      .requires_grad(requiresGrad))};
}

bool Tensor::gpuAvailable()
{
    return torch::cuda::is_available();
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

    if ((device == dtypes::kGPU) && !gpuAvailable())
    {
        throw std::runtime_error("trying to move tensor to GPU but none ara available!!");
    }

    _tensor = _tensor.to(dtypes::deviceTypeMap(device));
    _device = device;
    return *this;
}

Tensor &Tensor::requiresGrad(bool reqGrad)
{
    NT_PROFILE();

    _tensor = _tensor.set_requires_grad(reqGrad);
    _requiresGrad = reqGrad;
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

    return {_tensor.index(convertIndices(indices))};
}

Tensor::variantType Tensor::getVariantValue(const std::vector<int> &indices) const
{
    NT_PROFILE();

    switch (_dType)
    {
    case dtypes::kFloat:
        return _tensor.index(convertIndices(indices)).item<float>();

    case dtypes::kDouble:
        return _tensor.index(convertIndices(indices)).item<double>();

    case dtypes::kComplexFloat:
        return (std::complex<float>)_tensor.index(convertIndices(indices)).item<c10::complex<float>>();

    case dtypes::kComplexDouble:
        return (std::complex<double>)_tensor.index(convertIndices(indices)).item<c10::complex<double>>();

    // in theory this is not reachable so exclude it from code coverage
    // LCOV_EXCL_START
    default:
        NT_ERROR("Invalid dtype has been set for this tensor: {}", (int)_dType);
        NT_ERROR("{}:{}", __FILE__, __LINE__);
        throw;
    }
    // LCOV_EXCL_STOP
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

std::vector<long int> Tensor::getShape() const
{
    NT_PROFILE();

    return _tensor.sizes().vec();
}

Tensor Tensor::matmul(const Tensor &tensor1, const Tensor &tensor2)
{
    NT_PROFILE();

    return {torch::matmul(tensor1._tensor, tensor2._tensor)};
}

Tensor Tensor::outer(const Tensor &tensor1, const Tensor &tensor2)
{
    NT_PROFILE();

    return {torch::outer(tensor1._tensor, tensor2._tensor)};
}

Tensor Tensor::mul(const Tensor &tensor1, const Tensor &tensor2)
{
    NT_PROFILE();

    return {torch::mul(tensor1._tensor, tensor2._tensor)};
}

Tensor Tensor::add(const Tensor &tensor1, const Tensor &tensor2)
{
    NT_PROFILE();

    return {torch::add(tensor1._tensor, tensor2._tensor)};
}

Tensor Tensor::div(const Tensor &tensor1, const Tensor &tensor2)
{
    NT_PROFILE();

    return {torch::div(tensor1._tensor, tensor2._tensor)};
}

Tensor Tensor::pow(const Tensor &tensor, float scalar)
{
    NT_PROFILE();

    return {torch::pow(tensor._tensor, scalar)};
}

Tensor Tensor::pow(const Tensor &tensor, double scalar)
{
    NT_PROFILE();

    return {torch::pow(tensor._tensor, scalar)};
}

Tensor Tensor::pow(const Tensor &tensor, std::complex<float> scalar)
{
    NT_PROFILE();

    assert(tensor._dType == dtypes::kComplexFloat | tensor._dType == dtypes::kComplexDouble);

    return {torch::pow(tensor._tensor, c10::complex<float>(scalar.real(), scalar.imag()))};
}

Tensor Tensor::pow(const Tensor &tensor, std::complex<double> scalar)
{
    NT_PROFILE();

    assert(tensor._dType == dtypes::kComplexFloat | tensor._dType == dtypes::kComplexDouble);

    return {torch::pow(tensor._tensor, c10::complex<double>(scalar.real(), scalar.imag()))};
}

Tensor Tensor::sqrt(const Tensor &tensor)
{
    NT_PROFILE();

    return {torch::sqrt(tensor._tensor)};
}

Tensor Tensor::exp(const Tensor &tensor)
{
    NT_PROFILE();

    return {torch::exp(tensor._tensor)};
}

Tensor Tensor::transpose(const Tensor &tensor, int dim0, int dim1)
{
    NT_PROFILE();

    return {torch::transpose(tensor._tensor, dim0, dim1)};
}

Tensor Tensor::scale(const Tensor &tensor, float scalar)
{
    NT_PROFILE();

    return {torch::multiply(tensor._tensor, scalar)};
}

Tensor Tensor::scale(const Tensor &tensor, double scalar)
{
    NT_PROFILE();

    return {torch::multiply(tensor._tensor, scalar)};
}

Tensor Tensor::scale(const Tensor &tensor, std::complex<float> scalar)
{
    NT_PROFILE();

    assert(tensor._dType == dtypes::kComplexFloat | tensor._dType == dtypes::kComplexDouble);

    return {torch::multiply(tensor._tensor, c10::complex<float>(scalar.real(), scalar.imag()))};
}

Tensor Tensor::scale(const Tensor &tensor, std::complex<double> scalar)
{
    NT_PROFILE();

    assert(tensor._dType == dtypes::kComplexFloat | tensor._dType == dtypes::kComplexDouble);

    return {torch::multiply(tensor._tensor, c10::complex<double>(scalar.real(), scalar.imag()))};
}

void Tensor::matmul_(const Tensor &tensor2)
{
    NT_PROFILE();

    _tensor = torch::matmul(_tensor, tensor2._tensor);
}

void Tensor::mul_(const Tensor &tensor2)
{
    NT_PROFILE();

    _tensor = torch::mul(_tensor, tensor2._tensor);
}

void Tensor::div_(const Tensor &tensor2)
{
    NT_PROFILE();

    _tensor = torch::div(_tensor, tensor2._tensor);
}

void Tensor::scale_(float scalar)
{
    NT_PROFILE();

    _tensor = torch::multiply(_tensor, scalar);
}

void Tensor::scale_(std::complex<float> scalar)
{
    NT_PROFILE();

    _tensor = torch::multiply(_tensor, c10::complex<float>(scalar.real(), scalar.imag()));
}

void Tensor::pow_(float scalar)
{
    NT_PROFILE();

    _tensor = torch::pow(_tensor, scalar);
}

void Tensor::pow_(std::complex<float> scalar)
{
    NT_PROFILE();

    _tensor = torch::pow(_tensor, c10::complex<float>(scalar.real(), scalar.imag()));
}

void Tensor::exp_()
{
    NT_PROFILE();

    _tensor = torch::exp(_tensor);
}

void Tensor::transpose_(int dim0, int dim1)
{
    NT_PROFILE();

    _tensor = torch::transpose(_tensor, dim0, dim1);
}

void Tensor::eig(const Tensor &tensor, Tensor &eVals, Tensor &eVecs)
{
    NT_PROFILE();

    auto ret = torch::linalg_eig(tensor._tensor);
    eVals.setTensor(std::get<0>(ret));
    eVecs.setTensor(std::get<1>(ret));
}

void Tensor::eigh(const Tensor &tensor, Tensor &eVals, Tensor &eVecs)
{
    NT_PROFILE();

    auto ret = torch::linalg_eigh(tensor._tensor);
    eVals.setTensor(std::get<0>(ret));
    eVecs.setTensor(std::get<1>(ret));
}

void Tensor::eigvals(const Tensor &tensor, Tensor &eVals)
{
    NT_PROFILE();

    eVals.setTensor(torch::linalg_eigvals(tensor._tensor));
}

void Tensor::eigvalsh(const Tensor &tensor, Tensor &eVals)
{
    NT_PROFILE();

    eVals.setTensor(torch::linalg_eigvalsh(tensor._tensor));
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

    return {at::imag(_tensor)};
}

Tensor Tensor::conj() const
{
    NT_PROFILE();

    // torch::conj() returns a view of the original tensor
    // I *think* that means that the tensor returned here will be pointing to the
    // same memory as the original one might need to be careful with this
    return {torch::conj(_tensor)};
}

Tensor Tensor::abs() const
{
    NT_PROFILE();

    return {torch::abs(_tensor)};
}

Tensor Tensor::angle() const
{
    NT_PROFILE();

    return {torch::angle(_tensor)};
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

    return {_tensor + rhs._tensor};
}

Tensor Tensor::operator+(double rhs) const
{
    NT_PROFILE();

    return {_tensor + rhs};
}

Tensor Tensor::operator-(const Tensor &rhs) const
{
    NT_PROFILE();

    return {_tensor - rhs._tensor};
}

Tensor Tensor::operator-(double rhs) const
{
    NT_PROFILE();

    return {_tensor - rhs};
}

Tensor Tensor::operator-() const
{
    NT_PROFILE();

    return {-_tensor};
}

Tensor Tensor::operator*(const Tensor &rhs) const
{
    NT_PROFILE();

    return {_tensor * rhs._tensor};
}

Tensor Tensor::operator*(double rhs) const
{
    NT_PROFILE();

    return {_tensor * rhs};
}

Tensor Tensor::operator/(const Tensor &rhs) const
{
    NT_PROFILE();

    return {_tensor / rhs._tensor};
}

Tensor Tensor::operator/(double rhs) const
{
    NT_PROFILE();

    return {_tensor / rhs};
}

Tensor Tensor::cumsum(int dim) const
{
    NT_PROFILE();

    return {torch::cumsum(_tensor, dim)};
}

Tensor Tensor::sum() const
{
    NT_PROFILE();

    return {_tensor.sum()};
}

Tensor Tensor::sum(const std::vector<long int> &dims) const
{
    NT_PROFILE();

    return {torch::sum(_tensor, torch::OptionalArrayRef<long int>(dims))};
}

void Tensor::backward() const
{
    NT_PROFILE();

    _tensor.backward({}, /*keep_graph=*/true);
}

void Tensor::zeroGrad()
{
    NT_PROFILE();

    _tensor.grad().zero_();
}

Tensor Tensor::grad() const
{
    NT_PROFILE();

    if (!_requiresGrad)
    {
        throw std::runtime_error("Trying to access gradient of a Tensor that does not have requiresGrad!!!");
    }

    return {_tensor.grad()};
}

Tensor Tensor::sin(const Tensor &tensor)
{
    NT_PROFILE();

    return {torch::sin(tensor._tensor)};
}

Tensor Tensor::cos(const Tensor &tensor)
{
    NT_PROFILE();

    return {torch::cos(tensor._tensor)};
}

std::string Tensor::toString() const
{
    NT_PROFILE();

    std::ostringstream stream;
    stream << _tensor;
    return stream.str();
}