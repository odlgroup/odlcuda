#pragma once

#ifdef UTILITIES_STATIC_DEFINE
#  define UTILITIES_EXPORT
#  define UTILITIES_NO_EXPORT
#else
#  ifndef UTILITIES_EXPORT
#    ifdef _WIN32
#      ifdef rlutils_EXPORTS
        /* We are building this library */
#        define UTILITIES_EXPORT __declspec(dllexport)
#      else
          /* We are using this library */
#        define UTILITIES_EXPORT __declspec(dllimport)
#      endif
#    else	
#	define UTILITIES_EXPORT
#    endif
#  endif
#endif