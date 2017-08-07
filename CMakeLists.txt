cmake_minimum_required(VERSION 2.8)
project( MSCKF )
find_package( OpenCV 2.4.11 REQUIRED )


include_directories( ${OpenCV_INCLUDE_DIRS} /home/anoop/eigen/ )

file(GLOB msckf
      "src/main.cpp"
      "src/MSCKF.h"
      "src/MSCKF.cpp"
      "src/MSCKFUpdate.cpp"
	    "src/math_tool.cpp"
	    "src/math_tool.h"
      "fast-cpp-csv-parser/csv.h"
      #"src/updatecode.cpp"
)


add_executable( msckf ${msckf} )
set_target_properties(msckf PROPERTIES COMPILE_FLAGS "-std=c++11 -w -g -O3") 
target_link_libraries( msckf ${OpenCV_LIBS} )






##adding c++11 support

#include( CheckCXXCompilerFlag )
#check_cxx_compiler_flag( "-std=c++11"   COMPILER_SUPPORTS_CXX11 )
#check_cxx_compiler_flag( "-std=c++0x"   COMPILER_SUPPORTS_CXX0X )
#if( COMPILER_SUPPORTS_CXX11 )
#    if( CMAKE_COMPILER_IS_GNUCXX )
#        set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11" )
#    else()
#        set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11" )
#    endif()
#elseif( COMPILER_SUPPORTS_CXX0X)
#    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x" )
#else()
#    # MSVC, On by default (if available)
#endif()
