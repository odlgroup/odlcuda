set(Headers
	Phantom.h
	Ellipse.h
	EigenUtils.h
	StandardPhantoms.h
	utilities_export.h)

set(Sources
	Phantom.cpp
	StandardPhantoms.cpp)

include_directories ("${PROJECT_SOURCE_DIR}")

add_library(odlutils SHARED ${Headers} ${Sources})