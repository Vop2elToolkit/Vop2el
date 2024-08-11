cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

project(Vop2elEnv)

set(third_party_dir ${CMAKE_BINARY_DIR}/third-party/opencv)
file(MAKE_DIRECTORY ${third_party_dir})

# Add opencv with ransac that return all inliers
include(ExternalProject)
ExternalProject_Add(opencv
    PREFIX ${third_party_dir}
    BINARY_DIR ${third_party_dir}/build/opencv
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third-party/opencv
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${third_party_dir}/install/opencv
               -DCMAKE_BUILD_TYPE=Release
               -DCMAKE_CXX_STANDARD=17
               -DBUILD_opencv_apps=OFF
               -DBUILD_JAVA=OFF
               -DBUILD_TESTS=OFF
               -DBUILD_opencv_dnn=OFF
               -DBUILD_opencv_java_bindings_generator=OFF
               -DBUILD_opencv_js_bindings_generator=OFF
               -DBUILD_opencv_objc_bindings_generator=OFF
               -DBUILD_opencv_ml=OFF
               -DBUILD_opencv_objdetect=OFF
               -DBUILD_opencv_python3=OFF
               -DBUILD_opencv_python_bindings_generator=OFF
               -DBUILD_opencv_python_tests=OFF
               -DBUILD_opencv_stitching=OFF
               -DBUILD_opencv_ts=OFF

    BUILD_COMMAND cmake --build . -j 4
)

# Add vop2el library and test
ExternalProject_Add(vop2el_lib
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/vop2el_src
    BINARY_DIR ${CMAKE_BINARY_DIR}/vop2el_lib
    CMAKE_ARGS -DOpenCV_DIR=${third_party_dir}/install/opencv/lib/cmake/opencv4
               -DBUILD_SHARED_LIBS=ON
               
    DEPENDS opencv
    BUILD_COMMAND cmake --build . -j 4
)