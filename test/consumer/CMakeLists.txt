cmake_minimum_required(VERSION 3.16)

project(hypervec_consumer LANGUAGES CXX)

find_package(hypervec 1.14 CONFIG REQUIRED)

add_executable(hypervec_consumer main.cpp)
target_link_libraries(hypervec_consumer PRIVATE hypervec::hypervec)

enable_testing()
add_test(NAME hypervec_consumer.smoke COMMAND hypervec_consumer)
