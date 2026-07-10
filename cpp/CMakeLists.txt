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
set(JETSON_PI_ROOT "" CACHE PATH "Jetson-PI llama.cpp fork SOURCE root (dev add_subdirectory path)")
set(JetsonPI_ROOT "" CACHE PATH "Jetson-PI INSTALL prefix (production find_package path)")
if(FLASHRT_CPP_WITH_CUDA_STAGING)
  find_package(CUDAToolkit REQUIRED)
endif()
if(FLASHRT_CPP_WITH_CUDA_KERNELS)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
endif()
if(FLASHRT_CPP_WITH_JETSON_PI AND NOT BUILD_TESTING)
  message(STATUS
    "FLASHRT_CPP_WITH_JETSON_PI is exercised through test targets; building anyway.")
endif()

# §5.2 production deploy: two integration routes.
#  (1) Dev: -DJETSON_PI_ROOT=<source-tree>  -> add_subdirectory (in-tree build,
#      the GPU-verified path). The provider links the in-tree jetson_pi_* targets.
#  (2) Prod: -DJetsonPI_ROOT=<install-prefix> (no JETSON_PI_ROOT) -> find_package
#      an INSTALLED Jetson-PI via cmake/FindJetsonPI.cmake; the provider links the
#      imported JetsonPI::jetson_pi_* targets. Used for `cmake --install` layouts.
# Exactly one of the two must be set. Configuring both is an error: selecting a
# route implicitly could hide a stale cache entry and link the wrong Jetson-PI.
# The provider's link expression below uses a variable so the two routes are
# symmetric and the dev path stays byte-identical to before.
set(_frt_jetson_pi_pi0  jetson_pi_pi0)
set(_frt_jetson_pi_llm  jetson_pi_llm)
set(_frt_jetson_pi_mllm jetson_pi_mllm)
if(FLASHRT_CPP_WITH_JETSON_PI)
  if(JETSON_PI_ROOT AND JetsonPI_ROOT)
    message(FATAL_ERROR
      "Set exactly one Jetson-PI integration root: "
      "-DJETSON_PI_ROOT=<source-tree> for the dev add_subdirectory route, or "
      "-DJetsonPI_ROOT=<install-prefix> for the production find_package route. "
      "Both are currently set.")
  endif()
  if(NOT JETSON_PI_ROOT AND NOT JetsonPI_ROOT)
    message(FATAL_ERROR
      "FLASHRT_CPP_WITH_JETSON_PI requires exactly one explicit integration "
      "root: -DJETSON_PI_ROOT=<source-tree> for dev, or "
      "-DJetsonPI_ROOT=<install-prefix> for production.")
  endif()
  if(JETSON_PI_ROOT AND EXISTS "${JETSON_PI_ROOT}/CMakeLists.txt")
    # (1) dev: add_subdirectory the source tree
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
    message(STATUS "Jetson-PI: dev add_subdirectory path (${JETSON_PI_ROOT})")
  elseif(JETSON_PI_ROOT)
    # JETSON_PI_ROOT was set but does not point at a Jetson-PI source tree.
    # Fail loudly rather than silently fall through to the prod find_package
    # path (which would then error with a confusing "Jetson-PI not found").
    message(FATAL_ERROR
      "JETSON_PI_ROOT is set to '${JETSON_PI_ROOT}' but it is not a Jetson-PI "
      "source tree (no CMakeLists.txt found there). Fix the path, or unset "
      "JETSON_PI_ROOT and use -DJetsonPI_ROOT=<install-prefix> for the "
      "production find_package route.")
  else()
    # (2) prod: find an installed Jetson-PI. Scope the module-path addition to
    # this find so FindJetsonPI.cmake does not leak into unrelated find_package
    # calls later in the same configure.
    list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../cmake)
    find_package(JetsonPI REQUIRED)
    list(POP_FRONT CMAKE_MODULE_PATH)
    set(_frt_jetson_pi_pi0  JetsonPI::jetson_pi_pi0)
    set(_frt_jetson_pi_llm  JetsonPI::jetson_pi_llm)
    set(_frt_jetson_pi_mllm JetsonPI::jetson_pi_mllm)
    message(STATUS "Jetson-PI: production find_package path (${JetsonPI_ROOT})")
  endif()
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
  providers/llama_cpp/src/checkpoint_identity.cpp
  providers/llama_cpp/src/pi0_runtime.cpp
  providers/llama_cpp/src/llm_runtime.cpp
  providers/llama_cpp/src/mllm_runtime.cpp)
target_link_libraries(flashrt_cpp_llama_cpp_provider
  PUBLIC flashrt_runtime)
target_include_directories(flashrt_cpp_llama_cpp_provider
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/providers/llama_cpp/include)
if(FLASHRT_CPP_WITH_JETSON_PI)
  target_sources(flashrt_cpp_llama_cpp_provider
    PRIVATE providers/llama_cpp/src/jetson_pi_engine.cpp
            providers/llama_cpp/src/jetson_pi_link_check.cpp)
  target_link_libraries(flashrt_cpp_llama_cpp_provider
    PRIVATE ${_frt_jetson_pi_pi0} ${_frt_jetson_pi_llm} ${_frt_jetson_pi_mllm})
  target_compile_definitions(flashrt_cpp_llama_cpp_provider
    PRIVATE FLASHRT_CPP_WITH_JETSON_PI=1)

  # SHARED variant for Python ctypes consumption (Phase 2). Same sources as the
  # STATIC provider minus the test-only link check; exports
  # frt_llama_cpp_default_engine_factory + frt_llama_cpp_pi0_runtime_open_with_engine_factory
  # for dlopen. Linux default visibility exposes the extern "C" symbols.
  add_library(flashrt_cpp_llama_cpp_provider_c SHARED
    providers/llama_cpp/src/checkpoint_identity.cpp
    providers/llama_cpp/src/pi0_runtime.cpp
    providers/llama_cpp/src/llm_runtime.cpp
    providers/llama_cpp/src/mllm_runtime.cpp
    providers/llama_cpp/src/jetson_pi_engine.cpp)
  target_link_libraries(flashrt_cpp_llama_cpp_provider_c
    PUBLIC flashrt_runtime ${_frt_jetson_pi_pi0} ${_frt_jetson_pi_llm} ${_frt_jetson_pi_mllm})
  target_include_directories(flashrt_cpp_llama_cpp_provider_c
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/providers/llama_cpp/include)
  target_compile_definitions(flashrt_cpp_llama_cpp_provider_c
    PRIVATE FLASHRT_CPP_WITH_JETSON_PI=1)
  set_target_properties(flashrt_cpp_llama_cpp_provider_c PROPERTIES
    POSITION_INDEPENDENT_CODE ON)

  # §5.2 production install: lay the SHARED provider lib so a deployed FlashRT
  # has the ctypes dlopen target on the install prefix. Dev add_subdirectory
  # builds are unaffected (install() is a no-op unless cmake --install runs).
  include(GNUInstallDirs)
  if(JetsonPI_ROOT)
    set_target_properties(flashrt_cpp_llama_cpp_provider_c PROPERTIES
      INSTALL_RPATH "$ORIGIN;${JetsonPI_LIBRARY_DIRS}")
  else()
    set_target_properties(flashrt_cpp_llama_cpp_provider_c PROPERTIES
      INSTALL_RPATH "$ORIGIN")
  endif()
  install(TARGETS flashrt_cpp_llama_cpp_provider_c flashrt_runtime
          LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
          RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
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

    if(JETSON_PI_ROOT AND EXISTS "${JETSON_PI_ROOT}/vendor")
      add_executable(test_llama_cpp_jetson_pi_engine
        tests/test_llama_cpp_jetson_pi_engine.cpp)
      target_link_libraries(test_llama_cpp_jetson_pi_engine
        PRIVATE flashrt_cpp_llama_cpp_provider ${_frt_jetson_pi_pi0})
      target_include_directories(test_llama_cpp_jetson_pi_engine
        PRIVATE ${JETSON_PI_ROOT}/vendor)   # stb_image.h (dev tree)
      add_test(NAME llama_cpp_jetson_pi_engine
        COMMAND test_llama_cpp_jetson_pi_engine)
    endif()

    add_executable(test_llama_cpp_jetson_pi_llm
      tests/test_llama_cpp_jetson_pi_llm.cpp)
    target_link_libraries(test_llama_cpp_jetson_pi_llm
      PRIVATE flashrt_cpp_llama_cpp_provider ${_frt_jetson_pi_llm})
    add_test(NAME llama_cpp_jetson_pi_llm
      COMMAND test_llama_cpp_jetson_pi_llm)

    add_executable(test_llama_cpp_jetson_pi_mllm
      tests/test_llama_cpp_jetson_pi_mllm.cpp)
    target_link_libraries(test_llama_cpp_jetson_pi_mllm
      PRIVATE flashrt_cpp_llama_cpp_provider ${_frt_jetson_pi_mllm})
    add_test(NAME llama_cpp_jetson_pi_mllm
      COMMAND test_llama_cpp_jetson_pi_mllm)

    # Numerical-parity tests (jetsonpi迁移.txt §14): FlashRT provider path vs
    # a DIRECT jetson_pi_* call, same inputs/backend. Proves the port/stage/
    # verb plumbing does not perturb outputs. Pi0 uses action tolerance;
    # LLM/MLLM use greedy sampling and exact generated-text equality.
    if(JETSON_PI_ROOT AND EXISTS "${JETSON_PI_ROOT}/vendor")
      add_executable(test_llama_cpp_jetson_pi_parity
        tests/test_llama_cpp_jetson_pi_parity.cpp)
      target_link_libraries(test_llama_cpp_jetson_pi_parity
        PRIVATE flashrt_cpp_llama_cpp_provider ${_frt_jetson_pi_pi0})
      target_include_directories(test_llama_cpp_jetson_pi_parity
        PRIVATE ${JETSON_PI_ROOT}/vendor)   # stb_image.h (dev tree)
      add_test(NAME llama_cpp_jetson_pi_parity
        COMMAND test_llama_cpp_jetson_pi_parity)
    endif()

      add_executable(test_llama_cpp_jetson_pi_llm_parity
        tests/test_llama_cpp_jetson_pi_llm_parity.cpp)
    target_link_libraries(test_llama_cpp_jetson_pi_llm_parity
      PRIVATE flashrt_cpp_llama_cpp_provider ${_frt_jetson_pi_llm})
      add_test(NAME llama_cpp_jetson_pi_llm_parity
        COMMAND test_llama_cpp_jetson_pi_llm_parity)

      add_executable(test_llama_cpp_jetson_pi_mllm_parity
        tests/test_llama_cpp_jetson_pi_mllm_parity.cpp)
      target_link_libraries(test_llama_cpp_jetson_pi_mllm_parity
        PRIVATE flashrt_cpp_llama_cpp_provider ${_frt_jetson_pi_mllm})
      add_test(NAME llama_cpp_jetson_pi_mllm_parity
        COMMAND test_llama_cpp_jetson_pi_mllm_parity)
  endif()
endif()
