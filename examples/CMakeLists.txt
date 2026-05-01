# Included only when ARNOLDI_BUILD_EXAMPLES=ON (see parent CMakeLists.txt).

if(ARNOLDI_BUILD_EXAMPLES)
    add_executable(example_symmetric_laplacian symmetric_laplacian.cpp)
    target_link_libraries(example_symmetric_laplacian PRIVATE arnoldi)

    # Class-based Arnoldi<Kind, Scalar> demos (one per kind)
    foreach(_demo IN ITEMS arnoldi_sym arnoldi_nonsym arnoldi_herm)
        add_executable(example_${_demo} ${_demo}.cpp)
        target_link_libraries(example_${_demo} PRIVATE arnoldi)
    endforeach()

    # C++ ports of the Fortran reference drivers, built on arnoldi::Arnoldi
    foreach(_demo IN ITEMS arnoldi_dsdrv1 arnoldi_dndrv1 arnoldi_dsdrv2 arnoldi_dndrv2
                          arnoldi_dsdrv4 arnoldi_dsdrv5 arnoldi_dsdrv6)
        add_executable(example_${_demo} ${_demo}.cpp)
        target_include_directories(example_${_demo} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
        target_link_libraries(example_${_demo} PRIVATE arnoldi)
    endforeach()

    # MPI parallel example (optional — only if MPI found)
    find_package(MPI COMPONENTS CXX QUIET)
    if(MPI_CXX_FOUND)
        add_executable(example_arnoldi_mpi arnoldi_mpi.cpp)
        target_link_libraries(example_arnoldi_mpi PRIVATE arnoldi MPI::MPI_CXX)
        message(STATUS "MPI example: example_arnoldi_mpi (mpirun -np N ./example_arnoldi_mpi)")
    endif()
endif()

# Performance + correctness benchmark against original ARPACK
if(ARNOLDI_BUILD_EXAMPLES AND ORIGINAL_ARPACK_LIB)
    enable_language(Fortran)
    add_executable(bench_arpack benchmark.cpp)
    target_include_directories(bench_arpack PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    if(TARGET BLAS::BLAS AND TARGET LAPACK::LAPACK)
        target_link_libraries(bench_arpack PRIVATE
            arnoldi ${ORIGINAL_ARPACK_LIB} LAPACK::LAPACK BLAS::BLAS)
    else()
        target_link_libraries(bench_arpack PRIVATE
            arnoldi ${ORIGINAL_ARPACK_LIB} ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})
    endif()
    target_link_libraries(bench_arpack PRIVATE ${CMAKE_Fortran_IMPLICIT_LINK_LIBRARIES})

    add_test(NAME arnoldi_vs_arpack COMMAND bench_arpack --warmup 3 --reps 5)
    set_tests_properties(arnoldi_vs_arpack PROPERTIES LABELS "regression;performance")
    message(STATUS "Original ARPACK comparison enabled (lib: ${ORIGINAL_ARPACK_LIB})")
endif()
