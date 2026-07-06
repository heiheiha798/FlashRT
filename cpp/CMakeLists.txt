# FlashRT native C++ runtime path.
#
# This tree is intentionally separate from runtime/:
#   runtime/ = stable C ABI hand-off (frt_runtime_export_v1)
#   cpp/     = native model runtime implementation layers
#
# Build:
#   cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
#   cmake --build cpp/build -j

cmake_minimum_required(VERSION 3.18)
project(flashrt_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
include(CTest)

option(FLASHRT_CPP_WITH_EXEC "Build/link the exec layer for native replay" ON)
option(FLASHRT_CPP_WITH_CUDA_STAGING "Enable conservative CUDA H2D/D2H modality staging" ON)
option(FLASHRT_CPP_WITH_CUDA_KERNELS "Enable CUDA modality kernels" ON)
option(FLASHRT_CPP_WITH_JETSON_PI "Build optional Jetson-PI llama.cpp/mtmd integration checks" OFF)
set(JETSON_PI_ROOT "" CACHE PATH "Jetson-PI llama.cpp fork root")
if(FLASHRT_CPP_WITH_CUDA_STAGING)
  find_package(CUDAToolkit REQUIRED)
endif()
if(FLASHRT_CPP_WITH_CUDA_KERNELS)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
endif()
if(FLASHRT_CPP_WITH_JETSON_PI AND NOT BUILD_TESTING)
  message(FATAL_ERROR
    "FLASHRT_CPP_WITH_JETSON_PI currently requires BUILD_TESTING=ON "
    "(the Jetson-PI integration is exercised through test targets).")
endif()
if(FLASHRT_CPP_WITH_JETSON_PI)
  if(NOT JETSON_PI_ROOT OR NOT EXISTS "${JETSON_PI_ROOT}/CMakeLists.txt")
    message(FATAL_ERROR "FLASHRT_CPP_WITH_JETSON_PI requires -DJETSON_PI_ROOT=/path/to/Jetson-PI")
  endif()
  # FORCE: as a subdirectory embedding, FlashRT must own these Jetson-PI build
  # options regardless of stale cache or conflicting -D flags from the caller,
  # so the "mtmd without the full tools tree" isolation holds.
  set(LLAMA_BUILD_COMMON ON CACHE BOOL "llama: build common utils library" FORCE)
  set(LLAMA_CURL OFF CACHE BOOL "llama: use libcurl to download model from an URL" FORCE)
  set(LLAMA_HTTPLIB OFF CACHE BOOL "llama: disable httplib (FlashRT integration needs no HTTP)" FORCE)
  set(LLAMA_BUILD_TESTS OFF CACHE BOOL "llama: build tests" FORCE)
  set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "llama: build tools" FORCE)
  set(JETSON_PI_BUILD_MTMD ON CACHE BOOL "Jetson-PI: build the mtmd library without the full tools tree" FORCE)
  set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "llama: build examples" FORCE)
  set(LLAMA_BUILD_SERVER OFF CACHE BOOL "llama: build server example" FORCE)
  set(LLAMA_TOOLS_INSTALL OFF CACHE BOOL "llama: install tools" FORCE)
  add_subdirectory("${JETSON_PI_ROOT}"
                   "${CMAKE_CURRENT_BINARY_DIR}/jetson-pi"
                   EXCLUDE_FROM_ALL)
endif()
if(FLASHRT_CPP_WITH_EXEC AND NOT TARGET flashrt_exec)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../exec
                   ${CMAKE_CURRENT_BINARY_DIR}/exec)
endif()
if(NOT TARGET flashrt_runtime)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../runtime
                   ${CMAKE_CURRENT_BINARY_DIR}/runtime)
endif()

add_library(flashrt_cpp INTERFACE)
target_include_directories(flashrt_cpp
  INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/runtime/include
            ${CMAKE_CURRENT_SOURCE_DIR}/../runtime/include
            ${CMAKE_CURRENT_SOURCE_DIR}/../exec/include)
if(FLASHRT_CPP_WITH_EXEC)
  target_link_libraries(flashrt_cpp INTERFACE flashrt_exec)
endif()

set(FLASHRT_CPP_MODALITY_SRCS
  modalities/src/types.cpp
  modalities/src/vision_cpu.cpp
  modalities/src/action_cpu.cpp)
if(FLASHRT_CPP_WITH_CUDA_KERNELS)
  list(APPEND FLASHRT_CPP_MODALITY_SRCS modalities/src/vision_cuda.cu)
endif()

add_library(flashrt_cpp_modalities STATIC ${FLASHRT_CPP_MODALITY_SRCS})
target_include_directories(flashrt_cpp_modalities
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/modalities/include)
if(FLASHRT_CPP_WITH_CUDA_STAGING)
  target_compile_definitions(flashrt_cpp_modalities
    PUBLIC FLASHRT_CPP_WITH_CUDA_STAGING=1)
  target_link_libraries(flashrt_cpp_modalities PUBLIC CUDA::cudart)
endif()
if(FLASHRT_CPP_WITH_CUDA_KERNELS)
  target_compile_definitions(flashrt_cpp_modalities
    PRIVATE FLASHRT_CPP_WITH_CUDA_KERNELS=1)
  target_link_libraries(flashrt_cpp_modalities PUBLIC CUDA::cudart)
  set_target_properties(flashrt_cpp_modalities PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON)
endif()

add_library(flashrt_cpp_vla INTERFACE)
target_include_directories(flashrt_cpp_vla
  INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/families/vla/include)
target_link_libraries(flashrt_cpp_vla
  INTERFACE flashrt_cpp_modalities flashrt_cpp)

add_library(flashrt_cpp_pi05 STATIC
  models/pi05/src/spec.cpp
  models/pi05/src/io.cpp
  models/pi05/src/runtime.cpp)
target_include_directories(flashrt_cpp_pi05
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/models/pi05/include)
target_link_libraries(flashrt_cpp_pi05
  PUBLIC flashrt_cpp_modalities flashrt_cpp_vla)

add_library(flashrt_cpp_pi05_c SHARED
  models/pi05/src/c_api.cpp
  models/pi05/src/model_runtime.cpp)
target_link_libraries(flashrt_cpp_pi05_c
  PUBLIC flashrt_cpp_pi05 flashrt_runtime)
target_include_directories(flashrt_cpp_pi05_c
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/models/pi05/include)

add_library(flashrt_cpp_llama_cpp_provider STATIC
  providers/llama_cpp/src/pi0_runtime.cpp
  providers/llama_cpp/src/llm_runtime.cpp)
target_link_libraries(flashrt_cpp_llama_cpp_provider
  PUBLIC flashrt_runtime)
target_include_directories(flashrt_cpp_llama_cpp_provider
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/providers/llama_cpp/include)
if(FLASHRT_CPP_WITH_JETSON_PI)
  target_sources(flashrt_cpp_llama_cpp_provider
    PRIVATE providers/llama_cpp/src/jetson_pi_engine.cpp
            providers/llama_cpp/src/jetson_pi_link_check.cpp)
  target_link_libraries(flashrt_cpp_llama_cpp_provider
    PRIVATE jetson_pi_pi0 jetson_pi_llm)
  target_compile_definitions(flashrt_cpp_llama_cpp_provider
    PRIVATE FLASHRT_CPP_WITH_JETSON_PI=1)

  # SHARED variant for Python ctypes consumption (Phase 2). Same sources as the
  # STATIC provider minus the test-only link check; exports
  # frt_llama_cpp_default_engine_factory + frt_llama_cpp_pi0_runtime_open_with_engine_factory
  # for dlopen. Linux default visibility exposes the extern "C" symbols.
  add_library(flashrt_cpp_llama_cpp_provider_c SHARED
    providers/llama_cpp/src/pi0_runtime.cpp
    providers/llama_cpp/src/llm_runtime.cpp
    providers/llama_cpp/src/jetson_pi_engine.cpp)
  target_link_libraries(flashrt_cpp_llama_cpp_provider_c
    PUBLIC flashrt_runtime jetson_pi_pi0 jetson_pi_llm)
  target_include_directories(flashrt_cpp_llama_cpp_provider_c
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/providers/llama_cpp/include)
  target_compile_definitions(flashrt_cpp_llama_cpp_provider_c
    PRIVATE FLASHRT_CPP_WITH_JETSON_PI=1)
  set_target_properties(flashrt_cpp_llama_cpp_provider_c PROPERTIES
    POSITION_INDEPENDENT_CODE ON)
endif()

if(BUILD_TESTING)
  add_executable(test_cpp_modalities tests/test_modalities.cpp)
  target_link_libraries(test_cpp_modalities
    PRIVATE flashrt_cpp_pi05 flashrt_cpp_modalities)
  add_test(NAME cpp_modalities COMMAND test_cpp_modalities)

  if(FLASHRT_CPP_WITH_EXEC)
    add_executable(test_pi05_runtime tests/test_pi05_runtime.cpp)
    target_link_libraries(test_pi05_runtime
      PRIVATE flashrt_cpp_pi05 flashrt_cpp_modalities flashrt_exec)
    add_test(NAME pi05_runtime COMMAND test_pi05_runtime)
  endif()

  if(FLASHRT_CPP_WITH_EXEC AND FLASHRT_CPP_WITH_CUDA_STAGING)
    add_executable(test_device_staging tests/test_device_staging.cpp)
    target_link_libraries(test_device_staging
      PRIVATE flashrt_cpp_pi05 flashrt_cpp_modalities CUDA::cudart)
    add_test(NAME device_staging COMMAND test_device_staging)

    add_executable(test_pi05_c_api tests/test_pi05_c_api.cpp)
    target_link_libraries(test_pi05_c_api
      PRIVATE flashrt_cpp_pi05_c flashrt_exec CUDA::cudart)
    add_test(NAME pi05_c_api COMMAND test_pi05_c_api)

    add_executable(test_pi05_model_runtime tests/test_pi05_model_runtime.cpp)
    target_link_libraries(test_pi05_model_runtime
      PRIVATE flashrt_cpp_pi05_c flashrt_exec flashrt_runtime CUDA::cudart)
    add_test(NAME pi05_model_runtime COMMAND test_pi05_model_runtime)
  endif()

  add_executable(test_llama_cpp_provider tests/test_llama_cpp_provider.cpp)
  target_link_libraries(test_llama_cpp_provider
    PRIVATE flashrt_cpp_llama_cpp_provider flashrt_runtime)
  add_test(NAME llama_cpp_provider COMMAND test_llama_cpp_provider)

  if(FLASHRT_CPP_WITH_JETSON_PI)
    add_executable(test_llama_cpp_jetson_pi_link
      tests/test_llama_cpp_jetson_pi_link.cpp)
    target_link_libraries(test_llama_cpp_jetson_pi_link
      PRIVATE flashrt_cpp_llama_cpp_provider)
    add_test(NAME llama_cpp_jetson_pi_link
      COMMAND test_llama_cpp_jetson_pi_link)

    add_executable(test_llama_cpp_jetson_pi_engine
      tests/test_llama_cpp_jetson_pi_engine.cpp)
    target_link_libraries(test_llama_cpp_jetson_pi_engine
      PRIVATE flashrt_cpp_llama_cpp_provider jetson_pi_pi0)
    target_include_directories(test_llama_cpp_jetson_pi_engine
      PRIVATE ${JETSON_PI_ROOT}/vendor)   # stb_image.h for fixture loading
    add_test(NAME llama_cpp_jetson_pi_engine
      COMMAND test_llama_cpp_jetson_pi_engine)

    add_executable(test_llama_cpp_jetson_pi_llm
      tests/test_llama_cpp_jetson_pi_llm.cpp)
    target_link_libraries(test_llama_cpp_jetson_pi_llm
      PRIVATE flashrt_cpp_llama_cpp_provider jetson_pi_llm)
    add_test(NAME llama_cpp_jetson_pi_llm
      COMMAND test_llama_cpp_jetson_pi_llm)
  endif()
endif()
