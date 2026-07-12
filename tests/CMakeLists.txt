add_executable(test_small_instance test_small_instance.cpp)
target_link_libraries(test_small_instance PRIVATE tsp_core)
target_compile_definitions(test_small_instance PRIVATE TEST_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(test_parallel test_parallel.cpp)
target_link_libraries(test_parallel PRIVATE tsp_core)
target_compile_definitions(test_parallel PRIVATE TEST_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(test_cuda test_cuda.cpp)
target_link_libraries(test_cuda PRIVATE tsp_core)
target_compile_definitions(test_cuda PRIVATE TEST_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(test_cuda_candidate test_cuda_candidate.cpp)
target_link_libraries(test_cuda_candidate PRIVATE tsp_core)
target_compile_definitions(test_cuda_candidate PRIVATE TEST_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(test_cuda_qlsa_candidate test_cuda_qlsa_candidate.cpp)
target_link_libraries(test_cuda_qlsa_candidate PRIVATE tsp_core)
target_compile_definitions(test_cuda_qlsa_candidate PRIVATE TEST_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(test_qlsa_paper_lite test_qlsa_paper_lite.cpp)
target_link_libraries(test_qlsa_paper_lite PRIVATE tsp_core)
target_compile_definitions(test_qlsa_paper_lite PRIVATE TEST_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(test_island test_island.cpp)
target_link_libraries(test_island PRIVATE tsp_core)
target_compile_definitions(test_island PRIVATE TEST_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

if(TSP_ENABLE_OPENMP AND OpenMP_CXX_FOUND)
    target_link_libraries(test_small_instance PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(test_parallel PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(test_cuda PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(test_cuda_candidate PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(test_cuda_qlsa_candidate PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(test_qlsa_paper_lite PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(test_island PRIVATE OpenMP::OpenMP_CXX)
endif()

set(TSP_TEST_TARGETS
    test_small_instance
    test_parallel
    test_cuda
    test_cuda_candidate
    test_cuda_qlsa_candidate
    test_qlsa_paper_lite
    test_island
)

# C/C++ assert must remain active in every test configuration. CMake's
# optimized configurations define NDEBUG, so explicitly undefine it after
# the global configuration flags.
foreach(test_target IN LISTS TSP_TEST_TARGETS)
    if(MSVC)
        target_compile_options(${test_target} PRIVATE
            /UNDEBUG
            $<$<CONFIG:Release>:/O2>
        )
    else()
        target_compile_options(${test_target} PRIVATE
            -UNDEBUG
            $<$<CONFIG:Release>:-O3>
            $<$<CONFIG:Release>:-march=native>
        )
    endif()
endforeach()

add_test(
    NAME test_small_instance
    COMMAND ${CMAKE_COMMAND}
            -DTEST_STEM=test_small_instance
            -DBINARY_DIR=${CMAKE_BINARY_DIR}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/run_test_executable.cmake
)

add_test(
    NAME test_parallel
    COMMAND ${CMAKE_COMMAND}
            -DTEST_STEM=test_parallel
            -DBINARY_DIR=${CMAKE_BINARY_DIR}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/run_test_executable.cmake
)

add_test(
    NAME test_cuda
    COMMAND ${CMAKE_COMMAND}
            -DTEST_STEM=test_cuda
            -DBINARY_DIR=${CMAKE_BINARY_DIR}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/run_test_executable.cmake
)

add_test(
    NAME test_cuda_candidate
    COMMAND ${CMAKE_COMMAND}
            -DTEST_STEM=test_cuda_candidate
            -DBINARY_DIR=${CMAKE_BINARY_DIR}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/run_test_executable.cmake
)

add_test(
    NAME test_cuda_qlsa_candidate
    COMMAND ${CMAKE_COMMAND}
            -DTEST_STEM=test_cuda_qlsa_candidate
            -DBINARY_DIR=${CMAKE_BINARY_DIR}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/run_test_executable.cmake
)

add_test(
    NAME test_qlsa_paper_lite
    COMMAND ${CMAKE_COMMAND}
            -DTEST_STEM=test_qlsa_paper_lite
            -DBINARY_DIR=${CMAKE_BINARY_DIR}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/run_test_executable.cmake
)

add_test(
    NAME test_island
    COMMAND ${CMAKE_COMMAND}
            -DTEST_STEM=test_island
            -DBINARY_DIR=${CMAKE_BINARY_DIR}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/run_test_executable.cmake
)

find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
    add_test(
        NAME test_experiment_pipelines
        COMMAND ${Python3_EXECUTABLE} -S -m unittest -v
                tests.test_fair_experiment_pipeline
                tests.test_fair_runner_validation
                tests.test_fair_analyzer_integrity
                tests.test_island_ablation_pipeline
                tests.test_island_runner_hardening
                tests.test_island_analyzer_hardening
                tests.test_qlsa_variant_runner
    )
    set_tests_properties(test_experiment_pipelines PROPERTIES
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )
endif()
