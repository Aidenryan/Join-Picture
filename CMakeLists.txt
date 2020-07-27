cmake_minimum_required( VERSION 2.8 )
project( joint )

set( CMAKE_CXX_FLAGS "-std=c++11" )

find_package( OpenCV REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable( joint joint.cpp )

target_link_libraries( joint ${OpenCV_LIBS} )

add_executable( temp temp.cpp )

target_link_libraries( temp ${OpenCV_LIBS} )

add_executable( zhihu zhihu.cpp )

target_link_libraries( zhihu ${OpenCV_LIBS} )

add_executable( fe fe.cpp )

target_link_libraries( fe ${OpenCV_LIBS} )