cmake_minimum_required(VERSION 3.25)
project(keel CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

# -Wconversion is omitted deliberately: the IR indexes vectors with int ids
# that are range-validated at entry points; the flag is all noise here.
add_compile_options(-Wall -Wextra -Wpedantic -Wshadow)

include(CTest)
find_package(Python3 COMPONENTS Interpreter REQUIRED)

# Core library grows with each phase; only real (non-stub) sources are listed.
add_library(keel_core STATIC
  src/model.cpp
  src/checker.cpp
  src/oracle.cpp
  src/gen.cpp
  src/policy.cpp
  src/rt/ir.cpp
  src/rt/arena.cpp
  src/rt/executor.cpp
  src/rt/handover.cpp
  src/rt/bench_gen.cpp
)

add_executable(keel_bench src/rt/bench.cpp)
target_link_libraries(keel_bench PRIVATE keel_core)
target_include_directories(keel_core PUBLIC src external)

add_executable(keel_tests
  tests/test_smoke.cpp
  tests/test_model.cpp
  tests/test_checker.cpp
  tests/test_oracle.cpp
  tests/test_gen.cpp
  tests/test_policy.cpp
  tests/test_audit_repro.cpp
  tests/test_random_differential.cpp
  tests/test_rt_ir.cpp
  tests/test_rt_bench_gen.cpp
  tests/test_rt_arena.cpp
  tests/test_rt_executor.cpp
  tests/test_rt_handover.cpp
  tests/test_rt_adversarial.cpp
)
target_link_libraries(keel_tests PRIVATE keel_core)

add_executable(keel_experiment src/experiment.cpp)
target_link_libraries(keel_experiment PRIVATE keel_core)

add_executable(keel_domcheck src/domcheck.cpp)
target_link_libraries(keel_domcheck PRIVATE keel_core)

add_executable(keel_demo src/rt/demo.cpp)
target_link_libraries(keel_demo PRIVATE keel_core)

add_executable(keel_eval src/rt/eval.cpp)
target_link_libraries(keel_eval PRIVATE keel_core)

add_test(NAME keel_tests COMMAND keel_tests)
add_test(NAME evaluation_integrity
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/experiments/test_evaluation_integrity.py)
add_test(NAME evaluation_controls
  COMMAND ${Python3_EXECUTABLE}
          ${CMAKE_CURRENT_SOURCE_DIR}/experiments/test_evaluation_controls.py
          $<TARGET_FILE:keel_eval>)
