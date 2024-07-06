#pragma once

#include <nuTens/tensors/dtypes.hpp>
#include <map>
#include <any>
#include <iostream>

#if USE_PYTORCH
    #include <torch/torch.h>
#endif

class Tensor{

    /*!
    * @class Tensor
    * @brief Basic tensor class 
    * 
    * Tensor defines a basic interface for creating and manipulating tensors.
    * To create tensors you should use the Initialisers. These can be used on their own or chained together with the Setters to create the desired tensor.
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
        /// @name Initialisers
        /// Use these methods to initialise the tensor
        /// @{

        /// @brief Initialise this tensor with ones
        /// @arg length The length of the intitalised tensor
        /// @arg type The data type of the initialised tensor
        Tensor& ones(int length, NTdtypes::scalarType type, NTdtypes::deviceType device = NTdtypes::kCPU, bool requiresGrad = true);
        /// @brief Initialise this tensor with ones
        /// @arg shape The desired shape of the intitalised tensor
        /// @arg type The data type of the initialised tensor
        Tensor& ones(const std::vector<long int> &shape, NTdtypes::scalarType type, NTdtypes::deviceType device = NTdtypes::kCPU, bool requiresGrad = true);

        /// @brief Initialise this tensor with zeros
        /// @arg length The length of the intitalised tensor
        /// @arg type The data type of the initialised tensor
        Tensor &zeros(int length, NTdtypes::scalarType type, NTdtypes::deviceType device = NTdtypes::kCPU, bool requiresGrad = true);
        /// @brief Initialise this tensor with zeros
        /// @arg shape The desired shape of the intitalised tensor
        /// @arg type The data type of the initialised tensor
        Tensor &zeros(const std::vector<long int> &shape, NTdtypes::scalarType type, NTdtypes::deviceType device = NTdtypes::kCPU, bool requiresGrad = true);

        /// @}

        /// @name Setters
        /// @{
        /// @brief Set the underlying data type of this tensor
        Tensor& dType(NTdtypes::scalarType type);
        /// @brief Set the device that this tensor lives on
        Tensor& device(NTdtypes::deviceType device);
        /// @}

        /// @name Matrix Arithmetic
        /// Generally there are static functions with the pattern <function>(Mat1, Mat2) which will return a new matrix and inline equivalents with the pattern <function>_(Mat2) which will affect the object they are called by
        /// @{

        /// @brief Multiply two matrices together
        /// @arg t1 Left hand tensor
        /// @arg t2 Right hand tensor
        static Tensor matmul(const Tensor &t1, const Tensor &t2);
        
        /// @brief Scale a matrix by some scalar
        /// @arg s The scalar
        /// @arg t The tensor
        static Tensor scale(float s, const Tensor &t);
        /// @brief Scale a matrix by some complex scalar
        /// @arg s The scalar
        /// @arg t The tensor
        static Tensor scale(std::complex<float> s, const Tensor &t);

        /// @brief Inline matrix multiplication
        /// @arg t2 Right hand matrix to multiply with this one
        void matmul_(const Tensor &t2);

        
        /// @brief Inline matrix scaling
        /// @arg s The scalar
        void scale_(float s);
        /// @brief Inline complex matrix scaling
        /// @arg s The scalar
        void scale_(std::complex<float> s);

        /// @}

        /// @name Mathematical
        /// mathematical function overrides, generally work as expected, unless otherwise noted
        /// @{
        bool operator== (const Tensor &rhs) const;
        bool operator!= (const Tensor &rhs) const;
        Tensor operator+ (const Tensor &rhs) const;
        Tensor operator- (const Tensor &rhs) const;
        Tensor operator- () const;
        /// @}
        
        /// @brief Get the real part of a complex tensor
        Tensor real() const;
        /// @brief Get the imaginary part of a complex tensor
        Tensor imag() const;
        
        /// @brief Overwrite the << operator to print this tensor out to the command line
        friend std::ostream &operator<< (std::ostream& stream, const Tensor& tensor) {
            return stream << tensor.toString();
        };

        /// Print this object to a summary string 
        std::string toString() const ;

        /// @brief Set the value at a particular index of the tensor
        /// @arg indices The indices of the value to set
        /// @arg value The value to set it to
        void setValue(const Tensor &indices, const Tensor &value);
        void setValue(const std::vector<int> &indices, const Tensor &value);
        void setValue(const std::vector<int> &indices, float value);
        void setValue(const std::vector<int> &indices, std::complex<float> value);



        // Defining this here as it has to be in a header due to using template :(
#if USE_PYTORCH
        /// Get the value at a particular index of the tensor
        /// @arg indices The indices of the value to set
        template <typename T>
        inline T getValue(const std::vector<int> &indices){
            std::vector<at::indexing::TensorIndex> indicesVec;
            for(size_t i = 0; i < indices.size(); i++){
                indicesVec.push_back(at::indexing::TensorIndex(indices[i]));
            }

            return _tensor.index(indicesVec).item<T>();
        }
#endif
        
        /// Get the name of the backend library used to deal with tensors
        static std::string getTensorLibrary();
    
#if USE_PYTORCH
    public:
        inline const torch::Tensor &getTensor() const { return _tensor; }
    protected:
        torch::Tensor _tensor;
#endif
};