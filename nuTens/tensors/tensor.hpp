#pragma once

#include <any>
#include <complex>
#include <iostream>
#include <map>
#include <nuTens/tensors/dtypes.hpp>
#include <variant>
#include <vector>

#if USE_PYTORCH
#include <torch/torch.h>
#endif

class Tensor
{
    /*!
     * @class Tensor
     * @brief Basic tensor class
     *
     * Tensor defines a basic interface for creating and manipulating tensors.
     * To create tensors you should use the Initialisers. These can be used on
     * their own or chained together with the Setters to create the desired
     * tensor.
     *
     * For example
     * \code{.cpp}
     *   Tensor t;
     *   t.ones({3,3}).dType(NTdtypes::kFloat).device(NTdtypes::kGPU);
     * \endcode
     * will get you a 3x3 tensor of floats that lives on the GPU.
     * This is equivalent to
     * \code{.cpp}
     *   Tensor t;
     *   t.ones({3,3}, NTdtypes::kFloat, NTdtypes::kGPU);
     * \endcode
     */

  public:
    using indexType = std::variant<int, std::string>;

    /// @name Initialisers
    /// Use these methods to initialise the tensor
    /// @{

    /// @brief Initialise this tensor with ones
    /// @arg length The length of the intitalised tensor
    /// @arg type The data type of the initialised tensor
    Tensor &ones(int length, NTdtypes::scalarType type, NTdtypes::deviceType device = NTdtypes::kCPU,
                 bool requiresGrad = true);
    /// @brief Initialise this tensor with ones
    /// @arg shape The desired shape of the intitalised tensor
    /// @arg type The data type of the initialised tensor
    Tensor &ones(const std::vector<long int> &shape, NTdtypes::scalarType type,
                 NTdtypes::deviceType device = NTdtypes::kCPU, bool requiresGrad = true);

    /// @brief Initialise this tensor with zeros
    /// @arg length The length of the intitalised tensor
    /// @arg type The data type of the initialised tensor
    Tensor &zeros(int length, NTdtypes::scalarType type, NTdtypes::deviceType device = NTdtypes::kCPU,
                  bool requiresGrad = true);
    /// @brief Initialise this tensor with zeros
    /// @arg shape The desired shape of the intitalised tensor
    /// @arg type The data type of the initialised tensor
    Tensor &zeros(const std::vector<long int> &shape, NTdtypes::scalarType type,
                  NTdtypes::deviceType device = NTdtypes::kCPU, bool requiresGrad = true);

    /// @}

    /// @name Setters
    /// @{
    /// @brief Set the underlying data type of this tensor
    Tensor &dType(NTdtypes::scalarType type);
    /// @brief Set the device that this tensor lives on
    Tensor &device(NTdtypes::deviceType device);
    /// @brief Set whether the tensor requires a gradient
    Tensor &requiresGrad(bool reqGrad);
    /// @}

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
    /// @brief Scale a matrix by some complex scalar
    /// @arg s The scalar
    /// @arg t The tensor
    static Tensor scale(const Tensor &t, std::complex<float> s);

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

    /// @}

    /// @name Mathematical
    /// mathematical function overrides, generally work as expected, unless
    /// otherwise noted
    /// @{
    bool operator==(const Tensor &rhs) const;
    bool operator!=(const Tensor &rhs) const;
    Tensor operator+(const Tensor &rhs) const;
    Tensor operator-(const Tensor &rhs) const;
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

    /// @brief Get the result of summing this tensor over some dimension
    /// @param dim The dimension to sum over
    [[nodiscard]] Tensor cumsum(int dim) const;

    /// @brief Get the result of summing this tensor over all dimensions
    [[nodiscard]] Tensor sum() const;

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
    void setValue(const std::vector<std::variant<int, std::string>> &indices, const Tensor &value);
    void setValue(const std::vector<int> &indices, float value);
    void setValue(const std::vector<int> &indices, std::complex<float> value);

    /// @brief Get the value at a certain entry in the tensor
    /// @param indices The index of the entry to get
    [[nodiscard]] Tensor getValue(const std::vector<std::variant<int, std::string>> &indices) const;

    /// @brief Get the number of dimensions in the tensor
    [[nodiscard]] size_t getNdim() const;

    /// @brief Get the batch dimension size of the tensor
    [[nodiscard]] int getBatchDim() const;

    /// @brief Get the shape of the tensor
    [[nodiscard]] std::vector<int> getShape() const;

    // Defining this here as it has to be in a header due to using template :(
#if USE_PYTORCH
    /// @brief Get the value at a particular index of the tensor
    /// @arg indices The indices of the value to set
    template <typename T> inline T getValue(const std::vector<int> &indices)
    {
        std::vector<at::indexing::TensorIndex> indicesVec;
        indicesVec.reserve(indices.size());
        for (const int &i : indices)
        {
            indicesVec.push_back(at::indexing::TensorIndex(i));
        }

        return _tensor.index(indicesVec).item<T>();
    }

    /// Get the value of a size 0 tensor (scalar)
    template <typename T> inline T getValue()
    {
        return _tensor.item<T>();
    }
#endif

    /// Get the name of the backend library used to deal with tensors
    static std::string getTensorLibrary();

#if USE_PYTORCH
  public:
    [[nodiscard]] inline const torch::Tensor &getTensor() const
    {
        return _tensor;
    }

  private:
    torch::Tensor _tensor;
#endif
};