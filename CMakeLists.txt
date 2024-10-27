cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

project(Vop2elEnv)

option(BUILD_WITH_RERUN "Enable rerun build" ON)

# Create third-party build directory
set(third_party_dir ${CMAKE_BINARY_DIR}/third-party/)
file(MAKE_DIRECTORY ${third_party_dir})

# Add opencv with ransac that return all inliers
set(opencv_binary_dir ${third_party_dir}/opencv)
file(MAKE_DIRECTORY ${opencv_binary_dir})

include(ExternalProject)
ExternalProject_Add(opencv
    PREFIX ${opencv_binary_dir}
    BINARY_DIR ${opencv_binary_dir}/build/opencv
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third-party/opencv
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${opencv_binary_dir}/install/opencv
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

# Add rerun for visualization
if (BUILD_WITH_RERUN)
    set(rerun_src_dir ${CMAKE_CURRENT_SOURCE_DIR}/third-party/rerun)
    file(MAKE_DIRECTORY ${rerun_src_dir})

    set(rerun_binary_dir ${third_party_dir}/rerun)
    file(MAKE_DIRECTORY ${rerun_binary_dir})

    ExternalProject_Add(rerun
        PREFIX ${rerun_binary_dir}
        URL https://github.com/rerun-io/rerun/releases/download/0.17.0/rerun_cpp_sdk.zip
        DOWNLOAD_DIR ${rerun_src_dir}
        CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${rerun_binary_dir}
        BINARY_DIR ${rerun_binary_dir}
        BUILD_COMMAND cmake --build . -j 4
    )
endif()

# Add vop2el library and test
set(vop2el_dependencies opencv)
if (BUILD_WITH_RERUN)
    list( APPEND vop2el_dependencies rerun)
endif()

ExternalProject_Add(vop2el_lib
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/vop2el_src
    BINARY_DIR ${CMAKE_BINARY_DIR}/vop2el_lib
    CMAKE_ARGS -DOpenCV_DIR=${opencv_binary_dir}/install/opencv/lib/cmake/opencv4
               -Drerun_sdk_DIR=${rerun_binary_dir}/lib/cmake/rerun_sdk/
               -DBUILD_SHARED_LIBS=ON
               -DENABLE_RERUN_VISUALIZATION=${BUILD_WITH_RERUN}
               
    DEPENDS ${vop2el_dependencies}
    BUILD_COMMAND cmake --build . -j 4
)