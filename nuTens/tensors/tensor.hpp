#pragma once

#include <any>
#include <cassert>
#include <complex>
#include <iostream>
#include <map>
#include <nuTens/tensors/dtypes.hpp>
#include <nuTens/utils/instrumentation.hpp>
#include <nuTens/utils/logging.hpp>
#include <variant>
#include <vector>

#if USE_PYTORCH
#include <torch/torch.h>
#endif

/*!
 * @file tensor.hpp
 * @brief Defines the interface of a Tensor object
 */

namespace nuTens
{

class Tensor
{
    /*!
     * @class Tensor
     * @brief Basic tensor class
     *
     * Tensor defines a basic interface for creating and manipulating tensors.
     * To create tensors you should use the Constructors. These can be used on
     * their own or chained together with the Setters to create the desired
     * tensor.
     *
     * For example
     * \code{.cpp}
     *   Tensor = ones({3,3}).dType(dtypes::kFloat).device(dtypes::kGPU);
     * \endcode
     * will get you a 3x3 tensor of floats that lives on the GPU.
     * This is equivalent to
     * \code{.cpp}
     *   Tensor = ones({3,3}, dtypes::kFloat, dtypes::kGPU);
     * \endcode
     */

  public:
    /// Holds the possible "index" types, this allows us to pass integers OR strings as index values which allows us to
    /// do some basic slicing of tensors similar to python
    using indexType = std::variant<int, std::string>;

    /// Container that holds all allowed types that can be returned by a tensor
    using variantType = std::variant<int, float, double, std::complex<float>, std::complex<double>>;

    /// @name Constructors
    /// Use these methods to construct tensors
    /// @{

    /// @brief Default constructor with no initialisation
    Tensor() : _dType(dtypes::kUninitScalar), _device(dtypes::kUninitDevice) {};

    /// @brief Construct a 1-d array with specified values
    /// @arg values The values to include in the tensor
    Tensor(const std::vector<float> &values, dtypes::scalarType type = dtypes::kFloat,
           dtypes::deviceType device = dtypes::kCPU, bool requiresGrad = true);

    /// @brief Construct an identity tensor (has to be a 2d square tensor)
    /// @arg n The size of one of the sides of the tensor
    /// @arg type The data type of the tensor
    static Tensor eye(int n, dtypes::scalarType type = dtypes::kFloat, dtypes::deviceType device = dtypes::kCPU,
                      bool requiresGrad = true);

    /// @brief Construct a tensor with entries randomly initialised in the range [0, 1]
    /// @arg shape The desired shape of the intitalised tensor
    /// @arg type The data type of the tensor
    static Tensor rand(const std::vector<long int> &shape, dtypes::scalarType type = dtypes::kFloat,
                       dtypes::deviceType device = dtypes::kCPU, bool requiresGrad = true);

    /// @brief Construct a tensor diag values along the diagonal, and zero elsewhere
    /// @arg diag A 1-d tensor which represents the desired diagonal values
    static Tensor diag(const Tensor &diag);

    /// @brief Construct a tensor with ones
    /// @arg shape The desired shape of the intitalised tensor
    /// @arg type The data type of the tensor
    static Tensor ones(const std::vector<long int> &shape, dtypes::scalarType type = dtypes::kFloat,
                       dtypes::deviceType device = dtypes::kCPU, bool requiresGrad = true);

    /// @brief Construct a tensor with zeros
    /// @arg shape The desired shape of the intitalised tensor
    /// @arg type The data type of the tensor
    static Tensor zeros(const std::vector<long int> &shape, dtypes::scalarType type = dtypes::kFloat,
                        dtypes::deviceType device = dtypes::kCPU, bool requiresGrad = true);

    /// @}

    /// @name Setters
    /// @{
    /// @brief Set the underlying data type of this tensor
    Tensor &dType(dtypes::scalarType type);
    /// @brief Set the device that this tensor lives on
    Tensor &device(dtypes::deviceType device);
    /// @brief Set whether the tensor requires a gradient
    Tensor &requiresGrad(bool reqGrad);
    /// @brief Set whether or not the first dimension should be interpreted as a batch dimension
    inline Tensor &hasBatchDim(bool hasBatchDim)
    {
        _hasBatchDim = hasBatchDim;
        return *this;
    };
    /// @}

    /// @brief If the tensor does not already have a batch dimension (as set by hasBatchDim()) this will add one
    Tensor &addBatchDim();

    /// @brief add new dimension to the tensor at a particular index
    Tensor &unsqueeze(int index);

    /// @name Matrix Arithmetic
    /// Generally there are static functions with the pattern <function>(Mat1,
    /// Mat2) which will return a new matrix and inline equivalents with the
    /// pattern <function>_(Mat2) which will affect the object they are called by
    /// @{

    /// @brief Multiply two matrices together
    /// @arg t1 Left hand tensor
    /// @arg t2 Right hand tensor
    static Tensor matmul(const Tensor &t1, const Tensor &t2);

    /// @brief Outer product of two 1D tensors
    /// @arg t1 Left hand tensor
    /// @arg t2 Right hand tensor
    static Tensor outer(const Tensor &t1, const Tensor &t2);

    /// @brief Element-wise multiplication of two tensors
    /// @arg t1 Left hand tensor
    /// @arg t2 Right hand tensor
    static Tensor mul(const Tensor &t1, const Tensor &t2);

    /// @brief Element-wise division of two tensors
    /// @arg t1 Numerator
    /// @arg t2 Denominator
    static Tensor div(const Tensor &t1, const Tensor &t2);

    /// @brief Raise a matrix to a scalar power
    /// @arg t The tensor
    /// @arg s The scalar
    static Tensor pow(const Tensor &t, float s);
    /// @brief Raise a matrix to a scalar power
    /// @arg t The tensor
    /// @arg s The scalar
    static Tensor pow(const Tensor &t, std::complex<float> s);

    /// @brief Element-wise exponential
    /// @arg t The tensor
    static Tensor exp(const Tensor &t);

    /// @brief Get the transpose of a tensor
    /// @arg t The tensor
    /// @arg dim1 The first dimension to swap
    /// @arg dim2 The second dimension to swap
    static Tensor transpose(const Tensor &t, int dim1, int dim2);

    /// @brief Scale a matrix by some scalar
    /// @arg s The scalar
    /// @arg t The tensor
    static Tensor scale(const Tensor &t, float s);
    /// @brief Scale a matrix by some scalar
    /// @arg s The scalar
    /// @arg t The tensor
    static Tensor scale(const Tensor &t, double s);
    /// @brief Scale a matrix by some complex scalar
    /// @arg s The scalar
    /// @arg t The tensor
    static Tensor scale(const Tensor &t, std::complex<float> s);
    /// @brief Scale a matrix by some complex scalar
    /// @arg s The scalar
    /// @arg t The tensor
    static Tensor scale(const Tensor &t, std::complex<double> s);

    // ############################################
    // ################ Inlines ###################
    // ############################################

    /// @brief Inline matrix multiplication
    /// @arg t2 Right hand matrix to multiply with this one
    void matmul_(const Tensor &t2);

    /// @brief inline element-wise multiplication
    /// @arg t2 Right hand tensor
    void mul_(const Tensor &t2);

    /// @brief inline element-wise division
    /// @arg t2 Denominator
    void div_(const Tensor &t2);

    /// @brief Inline matrix scaling
    /// @arg s The scalar
    void scale_(float s);
    /// @brief Inline complex matrix scaling
    /// @arg s The scalar
    void scale_(std::complex<float> s);

    /// @brief Inline raise to scalar power
    /// @arg s The scalar
    void pow_(float s);
    /// @brief Inline raise to scalar power
    /// @arg s The scalar
    void pow_(std::complex<float> s);

    /// @brief Inline element-wise exponential
    void exp_();

    /// @brief Inline transpose
    /// @arg dim1 The first dimension to swap
    /// @arg dim2 The second dimension to swap
    void transpose_(int dim1, int dim2);

    /// @}

    /// @name Linear Algebra
    /// @{

    /// @brief Get eigenvalues and vectors of a tensor
    /// @arg t The tensor
    /// @param[out] eVals The eigenvalues
    /// @param[out] eVecs The eigenvectors
    static void eig(const Tensor &t, Tensor &eVals, Tensor &eVecs);

    /// @brief Get eigenvalues and vectors of a hermitian matrix
    /// @arg t The tensor
    /// @param[out] eVals The eigenvalues
    /// @param[out] eVecs The eigenvectors
    /// This is in general faster and more stable than @ref Tensor::eig
    /// and should be preferred in basically all cases where it can be used
    static void eigh(const Tensor &t, Tensor &eVals, Tensor &eVecs);

    /// @brief Get eigenvalues of a tensor
    /// @arg t The tensor
    /// @param[out] eVals The eigenvalues
    static void eigvals(const Tensor &t, Tensor &eVals);

    /// @brief Get eigenvalues of a hermitian matrix
    /// @arg t The tensor
    /// @param[out] eVals The eigenvalues
    /// This is in general faster and more stable than @ref Tensor::eigvals
    /// and should be preferred in basically all cases where it can be used
    static void eigvalsh(const Tensor &t, Tensor &eVals);

    /// @brief Perform QR decomposition on a hermitian matrix
    /// @arg t The tensor
    /// @param[out] Q
    /// @param[out] R
    static void qr(const Tensor &t, Tensor &Q, Tensor &R);

    /// @}

    /// @name Mathematical
    /// mathematical function overrides, generally work as expected, unless
    /// otherwise noted
    /// @{
    bool operator==(const Tensor &rhs) const;
    bool operator!=(const Tensor &rhs) const;
    Tensor operator+(const Tensor &rhs) const;
    Tensor operator-(const Tensor &rhs) const;
    Tensor operator+(double rhs) const;
    Tensor operator-(double rhs) const;
    Tensor operator*(const Tensor &rhs) const;
    Tensor operator*(double rhs) const;
    Tensor operator/(const Tensor &rhs) const;
    Tensor operator/(double rhs) const;
    Tensor operator-() const;
    /// @}

    /// @brief Get the real part of a complex tensor
    [[nodiscard]] Tensor real() const;
    /// @brief Get the imaginary part of a complex tensor
    [[nodiscard]] Tensor imag() const;
    /// @brief Get the complex conjugate of this tensor. If the underlying tensor
    /// is not complex, this will just return the tensor.
    [[nodiscard]] Tensor conj() const;
    /// @brief Get elementwise absolute magnitude of a complex tensor
    [[nodiscard]] Tensor abs() const;
    /// @brief Get elementwise phases of a complex tensor
    [[nodiscard]] Tensor angle() const;

    /// @brief Get the cumulative sum over some dimension
    /// @param dim The dimension to sum over
    [[nodiscard]] Tensor cumsum(int dim) const;

    /// @brief Get the result of summing this tensor over all dimensions
    [[nodiscard]] Tensor sum() const;

    /// @brief Get the result of summing this tensor over all dimensions
    /// @param dims The dimensions to sum over
    [[nodiscard]] Tensor sum(const std::vector<long int> &dims) const;

    /// @brief Get the cumulative sum over some dimension
    /// @param dim The dimension to sum over
    static inline Tensor cumsum(const Tensor &t, int dim)
    {
        return t.cumsum(dim);
    }

    /// @brief Get the result of summing this tensor over all dimensions
    static inline Tensor sum(const Tensor &t)
    {
        return t.sum();
    }

    /// @brief Get the result of summing this tensor over all dimensions
    /// @param dims The dimensions to sum over
    static inline Tensor sum(const Tensor &t, const std::vector<long int> &dims)
    {
        return t.sum(dims);
    }

    /// @name Gradients
    /// @{

    /// @brief Compute gradients of this tensor with respect to leaves
    /// Those can then be accessed using gradient()
    void backward() const;

    /// @brief Return a tensor containing the accumulated gradients calculated
    /// for this tensor after calling backward()
    [[nodiscard]] Tensor grad() const;

    /// @}

    /// @name Trigonometric
    /// @{

    /// @brief Get element-wise sin of a tensor
    /// @param t The tensor
    static Tensor sin(const Tensor &t);

    /// @brief Get element-wise cosine of a tensor
    /// @param t The tensor
    static Tensor cos(const Tensor &t);

    /// @}

    /// @brief Overwrite the << operator to print this tensor out to the command
    /// line
    friend std::ostream &operator<<(std::ostream &stream, const Tensor &tensor)
    {
        return stream << tensor.toString();
    };

    /// Print this object to a summary string
    [[nodiscard]] std::string toString() const;

    /// @brief Set the value at a particular index of the tensor
    /// @arg indices The indices of the value to set
    /// @arg value The value to set it to
    void setValue(const Tensor &indices, const Tensor &value);
    void setValue(const std::vector<indexType> &indices, const Tensor &value);
    void setValue(const std::vector<int> &indices, float value);
    void setValue(const std::vector<int> &indices, double value);
    void setValue(const std::vector<int> &indices, std::complex<float> value);
    void setValue(const std::vector<int> &indices, std::complex<double> value);

    /// @brief Get the value at a certain entry in the tensor
    /// @param indices The index of the entry to get
    [[nodiscard]] Tensor getValues(const std::vector<indexType> &indices) const;

    /// @brief Get the value at a certain entry in the tensor as an std::variant
    /// @details This mainly exists so we can get the values of a tensor in python as pybind11 DOES NOT like templated
    /// functions If using the c++ interface it is probably easier, faster and safer to use the templated getValue()
    /// function.
    [[nodiscard]] variantType getVariantValue(const std::vector<int> &indices) const;

    /// @brief Get the number of dimensions in the tensor
    [[nodiscard]] size_t getNdim() const;

    /// @brief Get the size of the batch dimension of the tensor
    [[nodiscard]] int getBatchDim() const;

    /// @brief Check if tensor has batch dimension prepended
    [[nodiscard]] bool getHasBatchDim() const;

    /// @brief Get the shape of the tensor
    [[nodiscard]] std::vector<int> getShape() const;

    /// Get the name of the backend library used to deal with tensors
    static std::string getTensorLibrary();

  protected:
    bool _hasBatchDim = false;
    dtypes::scalarType _dType;
    dtypes::deviceType _device;

    // ###################################################
    // ########## Tensor library specific stuff ##########
    // ###################################################

    // Defining this here as it has to be in a header due to using template :(
#if USE_PYTORCH
  public:
    /// @brief Get the value at a particular index of the tensor
    /// @arg indices The indices of the value to set
    template <typename T> inline T getValue(const std::vector<int> &indices) const
    {
        NT_PROFILE();

        return _tensor.index(convertIndices(indices)).item<T>();
    }

    /// Get the value of a size 0 tensor (scalar)
    template <typename T> inline T getValue() const
    {
        NT_PROFILE();

        return _tensor.item<T>();
    }

    // return the underlying tensor with type determined by the backend library
    [[nodiscard]] inline const torch::Tensor &getTensor() const
    {
        NT_PROFILE();

        return _tensor;
    }

    /// Builds a nuTens tensor from a torch tensor. Only available if using the
    /// pytorch backend... obviously
    static inline Tensor fromTorchTensor(const torch::Tensor &tensor)
    {

        NT_PROFILE();

        Tensor ret;
        ret.setTensor(tensor);

        return ret;
    }

  protected:
    /// Set the underlying tensor, setting the relevant information like _dtype and _device
    inline void setTensor(const torch::Tensor &tensor)
    {
        NT_PROFILE();

        _tensor = tensor;
        _dType = dtypes::invScalarTypeMap(tensor.scalar_type());
        _device = dtypes::invDeviceTypeMap(tensor.device().type());
    }

    /// Utility function to convert from a vector of ints to a vector of a10 tensor indices, which is needed for
    /// accessing values of a tensor.
    [[nodiscard]] static inline std::vector<at::indexing::TensorIndex> convertIndices(const std::vector<int> &indices)
    {
        NT_PROFILE();

        std::vector<at::indexing::TensorIndex> indicesVec;
        indicesVec.reserve(indices.size());
        for (const int &i : indices)
        {
            indicesVec.push_back(at::indexing::TensorIndex(i));
        }

        return indicesVec;
    }

    /// Utility function to convert from a vector of ints to a vector of a10 tensor indices, which is needed for
    /// accessing values of a tensor.
    [[nodiscard]] static inline std::vector<at::indexing::TensorIndex> convertIndices(
        const std::vector<Tensor::indexType> &indices)
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

        return indicesVec;
    }

  protected:
    torch::Tensor _tensor;
#endif
};

#if USE_PYTORCH
/// @brief Tensors are designed to be used when you want to frequently
///        access single elements.
/// @tparam Tdtype The data type the tensor will hold.
/// @tparam TnDims The number of dimensions of the tensor
/// @tparam Tdevice The device the tensor will live on
///
/// AccessedTensors are designed to be used for fast direct access to
/// the individual values of the underlying tensor. This speed comes at
/// the cost of some flexibility as you must define the type, number of
/// dimensions, and device at compile time. This allows us to use pyTorchs
/// tensor accessors to increase the access speed.
/// You should only use these when you intend to directly manipulate the
/// entries of the tensor. e.g. to set parameter values, or energy values
/// at the start of a computational chain.
template <typename Tdtype, int TnDims, dtypes::deviceType Tdevice> class AccessedTensor : public Tensor
{

  public:
    inline AccessedTensor(const Tensor &tensor) : AccessedTensor(tensor.getTensor()) {};

  private:
    AccessedTensor(const torch::Tensor &tensor)
        : _packedAccessor(tensor.packed_accessor32<Tdtype, TnDims>()), _accessor(tensor.accessor<Tdtype, TnDims>())
    {
        setTensor(tensor);
    };

  public:
    /// @name Setters
    /// dtype for accessedtensor is fixed at compile time
    inline AccessedTensor &dType(dtypes::scalarType type) = delete;

    /// device for accessedtensor is fixed at compile time
    inline AccessedTensor &device(dtypes::deviceType device) = delete;

    /// @brief Set whether the tensor requires a gradient
    inline AccessedTensor &requiresGrad(bool reqGrad)
    {
        NT_PROFILE();

        _tensor = _tensor.set_requires_grad(reqGrad);
        return *this;
    }

    /// @brief Set whether or not the first dimension should be interpreted as a batch dimension
    inline AccessedTensor &hasBatchDim(bool hasBatchDim)
    {
        _hasBatchDim = hasBatchDim;
        return *this;
    };

    /// @}

    /// n dimensions for accessedtensor is fixed at compile time
    inline AccessedTensor &addBatchDim() = delete;

    /// n dimensions for accessedtensor is fixed at compile time
    inline AccessedTensor &unsqueeze(int index) = delete;

    /// @brief Construct an identity tensor (has to be a 2d square tensor)
    /// @arg n The size of one of the sides of the tensor
    /// @arg type The data type of the tensor
    static AccessedTensor eye(bool requiresGrad = true)
    {

        NT_PROFILE();

        AccessedTensor<Tdtype, TnDims, Tdevice> ret(
            torch::eye(TnDims, torch::TensorOptions()
                                   .dtype(dtypes::scalarTypeMap(dtypes::scalarTypeFromRaw<Tdtype>()))
                                   .device(dtypes::deviceTypeMap(Tdevice))
                                   .requires_grad(requiresGrad)));

        ret._dType = dtypes::scalarTypeFromRaw<Tdtype>();
        ret._device = Tdevice;

        return ret;
    }

    /// @brief Construct a tensor with entries randomly initialised in the range [0, 1]
    /// @arg shape The desired shape of the intitalised tensor
    /// @arg type The data type of the tensor
    static AccessedTensor rand(const std::vector<long int> &shape, bool requiresGrad = true)
    {

        NT_PROFILE();

        assert((shape.size() == TnDims) && "dimensions in shape must match templated TnDims");

        AccessedTensor<Tdtype, TnDims, Tdevice> ret(
            torch::rand(c10::IntArrayRef(shape), torch::TensorOptions()
                                                     .dtype(dtypes::scalarTypeMap(dtypes::scalarTypeFromRaw<Tdtype>()))
                                                     .device(dtypes::deviceTypeMap(Tdevice))
                                                     .requires_grad(requiresGrad)));

        ret._dType = dtypes::scalarTypeFromRaw<Tdtype>();
        ret._device = Tdevice;

        return ret;
    }

    /// @brief Construct a tensor with ones
    /// @arg shape The desired shape of the intitalised tensor
    /// @arg type The data type of the tensor
    static AccessedTensor ones(const std::vector<long int> &shape, bool requiresGrad = true)
    {

        NT_PROFILE();

        assert((shape.size() == TnDims) && "dimensions in shape must match templated TnDims");

        AccessedTensor<Tdtype, TnDims, Tdevice> ret(
            torch::ones(c10::IntArrayRef(shape), torch::TensorOptions()
                                                     .dtype(dtypes::scalarTypeMap(dtypes::scalarTypeFromRaw<Tdtype>()))
                                                     .device(dtypes::deviceTypeMap(Tdevice))
                                                     .requires_grad(requiresGrad)));

        ret._dType = dtypes::scalarTypeFromRaw<Tdtype>();
        ret._device = Tdevice;
        return ret;
    }

    /// @brief Construct a tensor with zeros
    /// @arg shape The desired shape of the intitalised tensor
    /// @arg type The data type of the tensor
    static AccessedTensor zeros(const std::vector<long int> &shape, bool requiresGrad = true)
    {

        NT_PROFILE();

        assert((shape.size() == TnDims) && "dimensions in shape must match templated TnDims");

        torch::Tensor zeros =
            torch::zeros(c10::IntArrayRef(shape), torch::TensorOptions()
                                                      .dtype(dtypes::scalarTypeMap(dtypes::scalarTypeFromRaw<Tdtype>()))
                                                      .device(dtypes::deviceTypeMap(Tdevice))
                                                      .requires_grad(requiresGrad));

        AccessedTensor<Tdtype, TnDims, Tdevice> ret(zeros);

        ret._dType = dtypes::scalarTypeFromRaw<Tdtype>();
        ret._device = Tdevice;
        return ret;
    }

    /// @name Value Setters
    /// @{

    /// @brief Set a value in a 1D tensor
    void setValue(Tdtype value, int i)
    {

        NT_PROFILE();

        static_assert(TnDims == 1, "wrong number of indices");

        if (Tdevice == dtypes::kGPU)
        {
            _packedAccessor[i] = value;
        }

        else if (Tdevice == dtypes::kCPU)
        {
            _accessor[i] = value;
        }
    }

    /// @brief Set a value in a 2D tensor
    void setValue(Tdtype value, int i, int j)
    {

        NT_PROFILE();

        static_assert(TnDims == 2, "wrong number of indices");

        if (Tdevice == dtypes::kGPU)
        {
            _packedAccessor[i][j] = value;
        }

        else if (Tdevice == dtypes::kCPU)
        {
            _accessor[i][j] = value;
        }
    }

    /// @brief Set a value in a 3D tensor
    void setValue(Tdtype value, int i, int j, int k)
    {

        NT_PROFILE();

        static_assert(TnDims == 3, "wrong number of indices");

        if (Tdevice == dtypes::kGPU)
        {
            _packedAccessor[i][j][k] = value;
        }

        else if (Tdevice == dtypes::kCPU)
        {
            _accessor[i][j][k] = value;
        }
    }

    /// @}

    /// @name Value Getters
    /// @{

    /// @brief Get a value in a 1D tensor
    Tdtype getValue(int i) const
    {

        NT_PROFILE();

        static_assert(TnDims == 1, "wrong number of indices");

        if (Tdevice == dtypes::kGPU)
        {
            return _packedAccessor[i];
        }

        if (Tdevice == dtypes::kCPU)
        {
            return _accessor[i];
        }
    }

    /// @brief Get a value in a 2D tensor
    Tdtype getValue(int i, int j) const
    {

        NT_PROFILE();

        static_assert(TnDims == 2, "wrong number of indices");

        if (Tdevice == dtypes::kGPU)
        {
            return _packedAccessor[i][j];
        }

        if (Tdevice == dtypes::kCPU)
        {
            return _accessor[i][j];
        }
    }

    /// @brief Get a value in a 3D tensor
    Tdtype getValue(int i, int j, int k) const
    {

        NT_PROFILE();

        static_assert(TnDims == 3, "wrong number of indices");

        if (Tdevice == dtypes::kGPU)
        {
            return _packedAccessor[i][j][k];
        }

        if (Tdevice == dtypes::kCPU)
        {
            return _accessor[i][j][k];
        }
    }

    /// @}

    // only support up to 3D for now, should be enough for us

  private:
    torch::TensorAccessor<Tdtype, TnDims> _accessor;
    torch::PackedTensorAccessor32<Tdtype, TnDims> _packedAccessor;
};

// Specialisations of value getters for complex values
// needs to be specialised outside the body of the class
template <> inline std::complex<float> Tensor::getValue<std::complex<float>>(const std::vector<int> &indices) const
{
    NT_PROFILE();

    return static_cast<std::complex<float>>(_tensor.index(convertIndices(indices)).item<c10::complex<float>>());
}

template <> inline std::complex<double> Tensor::getValue<std::complex<double>>(const std::vector<int> &indices) const
{
    NT_PROFILE();

    return static_cast<std::complex<double>>(_tensor.index(convertIndices(indices)).item<c10::complex<double>>());
}

template <> inline std::complex<float> Tensor::getValue<std::complex<float>>() const
{
    NT_PROFILE();

    return static_cast<std::complex<float>>(_tensor.item<c10::complex<float>>());
}

template <> inline std::complex<double> Tensor::getValue<std::complex<double>>() const
{
    NT_PROFILE();

    return static_cast<std::complex<double>>(_tensor.item<c10::complex<double>>());
}

#endif

}; // namespace nuTens