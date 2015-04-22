RLcpp
=====

C++ backend for RL

Introduction
--------------------
This is a default backend for RL, intended to contain examples of RL interactions with C++.

It contains a default implementation of a CUDA based RN space with associated methods.

Building
--------------------
The project uses CMake to enable builds.

[Cmake webpage](http://www.cmake.org/)

#### Unix:
Start by going the the directory where you want your binaries and run

    ccmake PATH_TO_SOURCE
    make

To test the code run

    ./test/SimRec2DTest

#### Windows

To build on windows, open the CMake gui, run configure-generate and set the required variables. Then open the project with Visual Studio.

Code guidelines
--------------------
The code is written in C++11/14.

### Compilers
Currently tested to compile with VS2013, it is intended to be usable with all major compilers.

##### CUDA
The code uses C++11 features in device code, so CUDA 7.0 is required. CUDA 6.5 may work on some platforms (notably windows).

### Formating
The code is formatted using [CLang format](http://clang.llvm.org/docs/ClangFormat.html) as provided by the LLVM project. The particular style used is defined in the [formatting file](_clang-format).

External Dependences
--------------------
Current external dependencies are

#####Eigen 
A C++ template library for linear algebra: matrices, vectors, numerical solvers, and related algorithms. The version tested for is the dev-branch, but any 3.* should work.

[Eigen webpage](http://eigen.tuxfamily.org)

#####Boost 
General library with C++ code. This project specifically uses [Boost.Python](http://www.boost.org/doc/libs/1_58_0/libs/python/doc/index.html) to handle the python bindings.

[Boost webpage](http://www.boost.org/)
[Prebuilt boost (windows)](http://boost.teeks99.com/), this version uses python 2.7

#####CUDA
Used for GPU accelerated versions of algorithms

[CUDA](https://developer.nvidia.com/cuda-downloads)

### Included
These are distributed with this project.

#####Catch
A header only framework for automated testing. Currently shipped with the repository as a single include.

[Catch webpage](https://github.com/philsquared/Catch)
