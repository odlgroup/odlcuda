RLcpp
=====

C++ backend for RL

Introduction
--------------------
This is a default backend for RL, intended to contain examples of RL interactions with C++.

It contains a default implementation of a CUDA based RN space with associated methods.

Building
--------------------
The project uses CMake to enable builds. Using *cmake-gui* is recommended

[Cmake webpage](http://www.cmake.org/)

#### Unix:
Start by going the the directory where you want your binaries and run, 

    cmake-gui PATH_TO_SOURCE
    
And then set the required variables, to build, run
    
    make
    
To install the package to your python installation, run

    make PyInstall

#### Windows

To build on windows, open the CMake gui, run configure-generate and set the required variables. Then open the project with Visual Studio and build `PyInstall`.

Code guidelines
--------------------
The code is written in C++11/14.

### Compilers
The code is intended to be usable with all major compilers. Current status (2015-06-22) is

| Platform     	| Compiler 	| Cuda 	| Compute 	| Works 	|
|--------------	|----------	|------	|---------	|-------	|
| Windows 7    	| VS2013   	| 7.0  	| 2.0     	| ✔     	|
| Windows 7    	| VS2013   	| 7.0  	| 5.2     	| ✔     	|
| Windows 10   	| VS2015   	| 7.0  	| 5.2     	| TODO  	|
| Fedora 21    	| GCC 4.9  	| 7.0  	| 5.2     	| ✔     	|
| Ubuntu ?.? 	| ???      	| 7.0  	| 5.2     	| ✔     	|
| Mac OSX 	| ???      	| 7.0  	| 5.2     	| TODO     	|

### Formating
The code is formatted using [CLang format](http://clang.llvm.org/docs/ClangFormat.html) as provided by the LLVM project. The particular style used is defined in the [formatting file](_clang-format).

External Dependences
--------------------
Current external dependencies are

#####Python
The building block of RL, RLCpp needs acces to both python and numpy header files and compiled files to link against.

#####Eigen 
A C++ template library for linear algebra: matrices, vectors, numerical solvers, and related algorithms. The version tested for is the dev-branch, but any 3.* should work.

[Eigen webpage](http://eigen.tuxfamily.org)

#####Boost 
General library with C++ code. This project specifically uses [Boost.Python](http://www.boost.org/doc/libs/1_58_0/libs/python/doc/index.html) to handle the python bindings.

[Boost webpage](http://www.boost.org/)
[Prebuilt boost (windows)](http://boost.teeks99.com/), this version uses python 2.7

#####CUDA
Used for GPU accelerated versions of algorithms. The code uses C++11 features in device code, so CUDA 7.0 is required. CUDA 6.5 may work on some platforms (notably windows). It should as of now compile with compute capbability >2.0, in the future certain features require higher compute capability may be added.

[CUDA](https://developer.nvidia.com/cuda-downloads)

### Included
These are distributed with this project.

#####Catch
A header only framework for automated testing. Currently shipped with the repository as a single include.

[Catch webpage](https://github.com/philsquared/Catch)

Troublefinding
--------------
There are a few common errors encountered, this is the solution to some of these

## Installation
* If, when compiling, you get a error like
    
        NumpyConfig.h not found
    
    then it is likely that the variable `PYTHON_NUMPY_INCLUDE_DIR` is not correctly set.

* If, when compiling, you get an error that begins with
    
        [ 20%] Building NVCC (Device) object RLcpp/CMakeFiles/PyCuda.dir//./PyCuda_generated_cuda.cu.o /usr/include/c++/4.9.2/bits/alloc_traits.h(248): error: expected a ">"
    
    It may be that you are trying to compile with CUDA 6.5 and GCC 4.9, this combination is not supported by CUDA.

* If you get a error like
    
        Error	5	error LNK2019: unresolved external symbol "__declspec(dllimport) struct _object * __cdecl boost::python::detail::init_module(struct PyModuleDef &,void (__cdecl*)(void))" (__imp_?init_module@detail@python@boost@@YAPEAU_object@@AEAUPyModuleDef@@P6AXXZ@Z) referenced in function PyInit_PyUtils	C:\Programming\Projects\RLcpp_bin\RLcpp\utils.obj	PyUtils
    
    then it is likely that you are trying to build against unmatched python header files and boost python version

## Running

* If, when running the tests in python, you get an error like
    
        RuntimeError: function_attributes(): after cudaFuncGetAttributes: invalid device function
        
    It may be that the compute version used is not supported by your setup, try changing the cmake variable `CUDA_COMPUTE`.

* If, when running the test in python, you encounter an error like

        ImportError: No module named RLCpp
    
    It may be that you have not installed the package, run `make PyInstall` or equivalent.
