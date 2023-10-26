cmake_minimum_required(VERSION 3.0.0)
project(HDRSampling VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)
enable_testing()

find_package(OpenEXR REQUIRED)
include_directories(/opt/local/include/libomp)
LINK_DIRECTORIES(/opt/local/lib/libomp)

add_executable(HDRSampling main.cpp)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)


if(APPLE)
    if(CMAKE_C_COMPILER_ID MATCHES "Clang\$")
        set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp")
        set(OpenMP_C_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY omp)
        # set(OpenMP_LIBRARY omp)
    endif()

    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang\$")
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY omp)
        # set(OpenMP_LIBRARY omp)
    endif()

endif()

FIND_PACKAGE(OpenMP)
if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    # set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else()
    message("OPENMP NOT FOUND")

endif()
########################################

target_link_libraries(${PROJECT_NAME} OpenEXR::OpenEXR ${OpenMP_omp_LIBRARY})

# warning: Openmp and Eigen Conflict when compile flag cxx_std_14 or cxx_std_17, https://gitlab.com/libeigen/eigen/-/issues/1734
target_compile_features(HDRSampling PRIVATE cxx_std_11)
