cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

project(VOP2EL)

set(VOP2EL_SOURCES
    vop2el_lib/src/Vop2elAlgorithm.cpp
    vop2el_lib/src/StereoImagesHandler.cpp
    vop2el_lib/src/Vop2elMatcher.cpp
    vop2el_lib/src/PatchCorrector.cpp
    vop2el_lib/src/Common.cpp
)

add_library(Vop2elAlgorithm ${VOP2EL_SOURCES})
set_target_properties(Vop2elAlgorithm PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/build
)

target_include_directories(Vop2elAlgorithm PUBLIC vop2el_lib/include)

find_package(OpenMP REQUIRED)
target_link_libraries(Vop2elAlgorithm OpenMP::OpenMP_CXX)

find_package(Ceres REQUIRED)
target_link_libraries(Vop2elAlgorithm ceres)

find_package(Eigen3 REQUIRED NO_MODULE)
target_link_libraries(Vop2elAlgorithm Eigen3::Eigen)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(Vop2elAlgorithm ${OpenCV_LIBS})

install(DIRECTORY vop2el_lib/include DESTINATION ${CMAKE_BINARY_DIR}/install/include)

install(TARGETS Vop2elAlgorithm
        LIBRARY DESTINATION ${CMAKE_BINARY_DIR}/install/lib
        ARCHIVE DESTINATION ${CMAKE_BINARY_DIR}/install/lib
        RUNTIME DESTINATION ${CMAKE_BINARY_DIR}/install/bin
)

add_executable(Vop2elTester
            test/main.cpp
            test/Utils.cpp
)
set_target_properties(Vop2elTester PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../bin
)

target_include_directories(Vop2elTester PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/test)
target_include_directories(Vop2elTester PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/vop2el_lib/include)
target_link_libraries(Vop2elTester Vop2elAlgorithm)