cmake_minimum_required(VERSION 2.8)
project(CV_Mid_Test)
find_package(OpenCV REQUIRED)
set(HEADER_FILES 
    include/detection.h 
    include/utils.h
)
set(SOURCE_FILES 
    src/detection.cpp 
    src/utils.cpp
)
add_library(mylib ${SOURCE_FILES} ${HEADER_FILES})
target_include_directories(mylib 
    PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(mylib PRIVATE ${OpenCV_LIBS})
add_executable(detect src/main.cpp)
target_include_directories(detect PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(detect PRIVATE mylib)