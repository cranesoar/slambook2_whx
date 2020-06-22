cmake_minImum_required( VERSION 2.8 )

project( UseEigen )

add_executable( eigenMatrix eigenMatrix.cpp )

include_directories( "/usr/include/eigen3" )

set( CMAKE_BUILD_TYPE "Debug")   # 这句话注释掉好像也没影响debug
