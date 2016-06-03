if (WIN32)
	set(FILESTART "")
	set(FILETYPE "dll")
	set(TARGETFILETYPE "pyd")
	set(TARGET_PY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIGURATION>")
else(WIN32)
	set(FILESTART "lib")
	set(FILETYPE "so")
	set(TARGETFILETYPE "so")
	set(TARGET_PY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
endif(WIN32)

if(CUDA_ENABLED)
    set(PYTHON_LIBS ${PYTHON_LIBS} "odlpp_cuda")
endif(CUDA_ENABLED)

set(INSTALL_LIB "pyinstall")
add_custom_target(${INSTALL_LIB} COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py ${TARGET_PY}/__init__.py)
set_property(TARGET ${INSTALL_LIB} PROPERTY FOLDER python)

# copy plugin manager
add_custom_command(TARGET ${INSTALL_LIB} COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/odl_pluggin.py ${TARGET_PY}/odl_pluggin.py)
add_custom_command(TARGET ${INSTALL_LIB} COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/setup.py ${TARGET_PY}/setup.py)
add_custom_command(TARGET ${INSTALL_LIB} COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/cu_ntuples.py ${TARGET_PY}/cu_ntuples.py)
add_custom_command(TARGET ${INSTALL_LIB} COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/ufuncs.py ${TARGET_PY}/ufuncs.py)
MESSAGE( STATUS "GOT TO COPYING" )
MESSAGE( STATUS "${PYTHON_LIBS}")
# do the copying
foreach(file_i ${PYTHON_LIBS})
    MESSAGE( STATUS "${file_i}" )
    MESSAGE( STATUS "${TARGET_PY}/${FILESTART}${file_i}.${FILETYPE}" )
    add_custom_command(TARGET ${INSTALL_LIB} COMMAND ${CMAKE_COMMAND} -E copy ${TARGET_PY}/${FILESTART}${file_i}.${FILETYPE} ${TARGET_PY}/${file_i}.${TARGETFILETYPE})
endforeach()

add_custom_command(TARGET ${INSTALL_LIB} COMMAND cd ${TARGET_PY} && ${PYTHON_EXECUTABLE} setup.py install)

add_dependencies(${INSTALL_LIB} ${PYTHON_LIBS})

add_subdirectory(cuda)
