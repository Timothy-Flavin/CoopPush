#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <new> // Required for std::align_val_t
#ifdef _MSC_VER
#include <BaseTsd.h>
using ssize_t = SSIZE_T;
#endif

namespace py = pybind11;

// --- Define the alignment boundary ---
// Use the C++17 standard constant if available, otherwise default to 64 bytes.
#if defined(__cpp_lib_hardware_interference_size)
const size_t CACHE_LINE_ALIGNMENT = std::hardware_destructive_interference_size;
#else
const size_t CACHE_LINE_ALIGNMENT = 64;
#endif

/**
 * @brief Allocates a 1D NumPy array of doubles,
 * guaranteed to start on a cache-line boundary.
 * @param num_elements The number of doubles in the array.
 * @return A 1D py::array_t<double> (NumPy ndarray).
 */
py::array_t<double> create_aligned_double_buffer(ssize_t num_elements)
{
    const size_t n_bytes = num_elements * sizeof(double);
    if (n_bytes == 0)
    {
        return py::array_t<double>({num_elements});
    }

    // 1. Allocate aligned memory using C++17's aligned new.
    void *ptr = ::operator new(n_bytes, std::align_val_t(CACHE_LINE_ALIGNMENT));

    // 2. Create a "capsule" to manage the memory's lifetime.
    // This tells Python's GC how to free this *specific* aligned memory.
    py::capsule deleter(ptr, [](void *p)
                        { ::operator delete(p, std::align_val_t(CACHE_LINE_ALIGNMENT)); });

    // 3. Create the NumPy array wrapper (zero-copy).
    return py::array_t<double>(
        {num_elements},             // Shape (1D)
        {sizeof(double)},           // Strides
        static_cast<double *>(ptr), // Pointer to data
        deleter                     // "Base" object that owns the memory
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

// --- Pybind11 Module Definition ---
// (Replace 'my_allocator' with your desired module name)
PYBIND11_MODULE(aligned_numpy_allocator, m)
{
    m.doc() = "Module with cache-line-aligned memory allocators for NumPy";

    // You can also expose the alignment size to Python
    m.attr("CACHE_LINE_SIZE") = py::int_(CACHE_LINE_ALIGNMENT);

    m.def("create_aligned_double_buffer", &create_aligned_double_buffer,
          py::arg("num_elements"),
          "Creates a 1D NumPy array (float64) with cache-line alignment.");

    m.def("create_aligned_bool_buffer", &create_aligned_bool_buffer,
          py::arg("num_elements"),
          "Creates a 1D NumPy array (bool) with cache-line alignment.");
}