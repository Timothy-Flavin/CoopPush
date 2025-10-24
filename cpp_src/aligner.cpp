#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <new> // Required for std::align_val_t
#ifdef _MSC_VER
#include <BaseTsd.h>
using ssize_t = SSIZE_T;
#endif
#include "aligner.h"

namespace py = pybind11;

// --- Define the alignment boundary ---
// Use the C++17 standard constant if available, otherwise default to 64 bytes.
#if defined(__cpp_lib_hardware_interference_size)
const size_t CACHE_LINE_ALIGNMENT = std::hardware_destructive_interference_size;
#else
const size_t CACHE_LINE_ALIGNMENT = 64;
#endif

/**
 * @brief Allocates a 1D NumPy array of floats,
 * guaranteed to start on a cache-line boundary.
 * @param num_elements The number of floats in the array.
 * @return A 1D py::array_t<float> (NumPy ndarray).
 */
py::array_t<float> create_aligned_float_buffer(ssize_t num_elements)
{
    const size_t n_bytes = num_elements * sizeof(float);
    if (n_bytes == 0)
    {
        return py::array_t<float>({num_elements});
    }

    // 1. Allocate aligned memory using C++17's aligned new.
    void *ptr = ::operator new(n_bytes, std::align_val_t(CACHE_LINE_ALIGNMENT));

    // 2. Create a "capsule" to manage the memory's lifetime.
    // This tells Python's GC how to free this *specific* aligned memory.
    py::capsule deleter(ptr, [](void *p)
                        { ::operator delete(p, std::align_val_t(CACHE_LINE_ALIGNMENT)); });

    // 3. Create the NumPy array wrapper (zero-copy).
    return py::array_t<float>(
        {num_elements},            // Shape (1D)
        {sizeof(float)},           // Strides
        static_cast<float *>(ptr), // Pointer to data
        deleter                    // "Base" object that owns the memory
    );
}

/**
 * @brief Allocates a 1D NumPy array of bools,
 * guaranteed to start on a cache-line boundary.
 * @param num_elements The number of bools in the array.
 * @return A 1D py::array_t<bool> (NumPy ndarray).
 */
py::array_t<bool> create_aligned_bool_buffer(ssize_t num_elements)
{
    const size_t n_bytes = num_elements * sizeof(bool);
    if (n_bytes == 0)
    {
        return py::array_t<bool>({num_elements});
    }

    // 1. Allocate aligned memory.
    void *ptr = ::operator new(n_bytes, std::align_val_t(CACHE_LINE_ALIGNMENT));

    // 2. Create the capsule deleter.
    py::capsule deleter(ptr, [](void *p)
                        { ::operator delete(p, std::align_val_t(CACHE_LINE_ALIGNMENT)); });

    // 3. Create the NumPy array wrapper.
    return py::array_t<bool>(
        {num_elements},           // Shape (1D)
        {sizeof(bool)},           // Strides
        static_cast<bool *>(ptr), // Pointer to data
        deleter                   // "Base" object
    );
}

py::array_t<float> create_aligned_float_buffer_2d(ssize_t dim1, ssize_t dim2)
{

    ssize_t num_elements = dim1 * dim2;
    const size_t n_bytes = num_elements * sizeof(float);
    if (n_bytes == 0)
    {
        return py::array_t<float>({dim1, dim2});
    }

    // 1. Allocate aligned memory (same as before)
    void *ptr = ::operator new(n_bytes, std::align_val_t(CACHE_LINE_ALIGNMENT));

    // 2. Create deleter capsule (same as before)
    py::capsule deleter(ptr, [](void *p)
                        { ::operator delete(p, std::align_val_t(CACHE_LINE_ALIGNMENT)); });

    // 3. Create the 2D NumPy array wrapper
    //    This is the key change.
    return py::array_t<float>(
        {dim1, dim2},                          // Shape: {N_ENVS, obs_size}
        {dim2 * sizeof(float), sizeof(float)}, // Strides
        static_cast<float *>(ptr),             // Pointer to flat data
        deleter                                // "Base" object
    );
}
