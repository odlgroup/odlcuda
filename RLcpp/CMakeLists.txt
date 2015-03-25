include_directories ("${PROJECT_SOURCE_DIR}")

set(LIB_NAME "PyUtils")
set(PYTHON_LIBS ${PYTHON_LIBS} ${LIB_NAME})
add_library(${LIB_NAME} SHARED function_traits.h numpy_utils.h numpy_eigen.h wrap.h utils.cpp)
set_property(TARGET ${LIB_NAME} PROPERTY FOLDER python)
target_link_libraries(${LIB_NAME} ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})

if (CUDA_ENABLED)
	set(LIB_NAME "PyCuda")
	set(PYTHON_LIBS ${PYTHON_LIBS} ${LIB_NAME})
	cuda_include_directories ("${PROJECT_SOURCE_DIR}")
	CUDA_ADD_LIBRARY(${LIB_NAME} SHARED "cuda.cpp" "cuda.cu" "thrustUtils.h")
	# CUDA_ADD_CUBLAS_TO_TARGET( ${LIB_NAME} ) possibly needed
	set_property(TARGET ${LIB_NAME} PROPERTY FOLDER python)
	target_link_libraries(${LIB_NAME} ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})
endif()

#TODO: LINUX?
set(INSTALL_LIB "PyInstall")
set(TARGET_PY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIGURATION>")
add_custom_target(${INSTALL_LIB} COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py ${TARGET_PY}/__init__.py)
set_property(TARGET ${INSTALL_LIB} PROPERTY FOLDER python)

# do the copying
foreach(file_i ${PYTHON_LIBS})
    add_custom_command(TARGET ${INSTALL_LIB} COMMAND ${CMAKE_COMMAND} -E copy ${TARGET_PY}/${file_i}.dll ${TARGET_PY}/${file_i}.pyd)
endforeach()

add_custom_command(TARGET ${INSTALL_LIB} COMMAND cd ${TARGET_PY} && python ${CMAKE_CURRENT_SOURCE_DIR}/setup.py install)
									
add_dependencies(${INSTALL_LIB} ${PYTHON_LIBS})