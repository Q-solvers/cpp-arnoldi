# Test suite — built on Catch2 v3 (fetched at configure time).

include(FetchContent)
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.7.1
)
FetchContent_MakeAvailable(Catch2)

# Catch2 ships CMake helpers (catch_discover_tests) under extras/.
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(Catch)

foreach(_test IN ITEMS test_validation test_getv0 test_solver test_helpers test_modes test_low_level_validation)
    add_executable(${_test} ${_test}.cpp)
    target_link_libraries(${_test} PRIVATE arnoldi Catch2::Catch2WithMain)
    catch_discover_tests(${_test})
endforeach()

# CUDA-backend test. Mirrors test_solver but launches the matvec as a CUDA
# kernel and uses Arnoldi<..., CudaBackend>. Built only when ARNOLDI_USE_CUDA
# is ON; that option already enables CUDA::cudart and CUDA::cublas on the
# arnoldi interface target.
if(ARNOLDI_USE_CUDA)
    enable_language(CUDA)
    add_executable(test_solver_cuda test_solver_cuda.cu)
    set_target_properties(test_solver_cuda PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON)
    target_link_libraries(test_solver_cuda PRIVATE arnoldi Catch2::Catch2WithMain)
    catch_discover_tests(test_solver_cuda)
endif()
