cmake_minImum_required( VERSION 2.8 )

project( HelloSLAM )

add_executable( helloSLAM helloSLAM.cpp )

add_executable( useHello useHello.cpp )

target_link_libraries( useHello hello_shared )

add_library( hello libHelloSLAM.cpp )

add_library( hello_shared SHARED libHelloSLAM.cpp )

set( CMAKE_BUILD_TYPE "Debug")   # 这句话注释掉好像也没影响debug
