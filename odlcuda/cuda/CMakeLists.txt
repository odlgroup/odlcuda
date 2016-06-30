include_directories ("${PROJECT_SOURCE_DIR}")

if(CUDA_ENABLED)
    set(LIB_NAME "odlcuda_")
    cuda_include_directories ("${PROJECT_SOURCE_DIR}")
    CUDA_ADD_LIBRARY(${LIB_NAME} SHARED "cuda.cpp"
                                        "cuda.cu"
                                        "DeviceVector.h"
                                        "DeviceVectorImpl.h"
                                        "CudaVector.h"
                                        "CudaVectorImpl.h"
                                        "CudaVectorImpl.cu"
                                        "SliceHelper.h"
                                        "TypeMacro.h"
                                        "UFunc.h"
                                        "UFunc.cu"
                                        "Reduction.h"
                                        "Reduction.cu")
    # CUDA_ADD_CUBLAS_TO_TARGET( ${LIB_NAME} ) possibly needed
    set_property(TARGET ${LIB_NAME} PROPERTY FOLDER python)
    target_link_libraries(${LIB_NAME} ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})
endif(CUDA_ENABLED)

