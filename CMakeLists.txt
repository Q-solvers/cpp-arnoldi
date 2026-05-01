cmake_minimum_required(VERSION 3.16)
project(arnoldi_callback VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(ARNOLDI_BUILD_EXAMPLES "Build example programs" OFF)
option(ARNOLDI_REGRESSION "Build regression benchmark against original ARPACK (requires arpackng)" OFF)

find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

add_library(arnoldi INTERFACE)
add_library(arnoldi::arnoldi ALIAS arnoldi)

target_include_directories(arnoldi INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

if(TARGET BLAS::BLAS AND TARGET LAPACK::LAPACK)
    target_link_libraries(arnoldi INTERFACE BLAS::BLAS LAPACK::LAPACK)
else()
    target_link_libraries(arnoldi INTERFACE ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
endif()

include(GNUInstallDirs)
install(TARGETS arnoldi EXPORT arnoldiTargets)
install(DIRECTORY include/arnoldi DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT arnoldiTargets
    FILE arnoldiTargets.cmake
    NAMESPACE arnoldi::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/arnoldi
)

option(ARNOLDI_BUILD_TESTS "Build unit tests" OFF)

if(ARNOLDI_BUILD_EXAMPLES OR ARNOLDI_BUILD_TESTS)
    enable_testing()
endif()

if(ARNOLDI_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(ARNOLDI_BUILD_TESTS)
    add_subdirectory(tests)
endif()
