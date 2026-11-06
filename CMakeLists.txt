cmake_minimum_required(VERSION 3.23)
project(ov_llm_cli LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenVINOGenAI REQUIRED)

add_executable(ov_llm_cli src/main.cpp)

target_link_libraries(ov_llm_cli PRIVATE openvino::genai)
