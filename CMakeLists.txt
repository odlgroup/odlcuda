cmake_minimum_required (VERSION 2.6)

project (odlpp)

#Set binary dir
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)

# add cmake find modules
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

# Use folders in MSVC
set_property(GLOBAL PROPERTY USE_FOLDERS true)

# Create clang compile for YouCompleteMe
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Add test project
enable_testing ()
add_definitions(-DCTEST_OUTPUT_ON_FAILURE)

# Add eigen
find_package(Eigen3 REQUIRED)
include_directories(SYSTEM ${EIGEN3_INCLUDE_DIR})

# Add cuda
option(CUDA_ENABLED "Rnable CUDA" TRUE)
if(CUDA_ENABLED)
	find_package(CUDA QUIET REQUIRED)

	set(CUDA_COMPUTE "52" CACHE STRING "Cuda compute capability to compile for")

	# Pass options to NVCC
	set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_${CUDA_COMPUTE},code=sm_${CUDA_COMPUTE})

	# Rnable fast math
	option(CUDA_USE_FAST_MATH "Rnable fast math in cuda (can decrease precision)" TRUE)
	if(CUDA_USE_FAST_MATH)
		set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-use_fast_math)
	endif()
endif()

# Add openMP
option(OPENMP_ENABLED "Rnable OpenMP" TRUE)
if(OPENMP_ENABLED)
	find_package(OpenMP QUIET REQUIRED)
	set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# Rnable c++11
if (MSVC)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4251 /bigobj")
else()
	include(CheckCXXCompilerFlag)
	CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
	CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
	if(COMPILER_SUPPORTS_CXX11)
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
	elseif()
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
	else()
		message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
	endif()

	if (CUDA_ENABLED)
		set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -std=c++11)
		SET(CUDA_PROPAGATE_HOST_FLAGS OFF)
	endif()
endif()

# Python bindings
option(PYTHON_ENABLED "Create python bindings" TRUE)

if(PYTHON_ENABLED)
	set(Boost_USE_SHARED_LIBS   ON)
	set(Boost_USE_MULTITHREADED ON)
	find_package(Boost COMPONENTS
                 python
                 REQUIRED)
	include_directories(${Boost_INCLUDE_DIRS})
	link_directories(${Boost_LIBRARY_DIRS})

	# Find python
	find_package(PythonLibs REQUIRED)
	include_directories(${PYTHON_INCLUDE_DIRS})

	# Find numpy
	find_package(Numpy REQUIRED)
	include_directories(${PYTHON_NUMPY_INCLUDE_DIR})

	link_directories(${PYTHON_LIBRARIES})
endif(PYTHON_ENABLED)

# Add sub directories as needed
add_subdirectory (ODLpp)
add_subdirectory (LCRUtils)
