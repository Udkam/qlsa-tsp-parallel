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

if(TSP_ENABLE_OPENMP AND OpenMP_CXX_FOUND)
    target_link_libraries(test_small_instance PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(test_parallel PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(test_cuda PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(test_cuda_candidate PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(test_cuda_qlsa_candidate PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(test_qlsa_paper_lite PRIVATE OpenMP::OpenMP_CXX)
endif()

if(MSVC)
    target_compile_options(test_small_instance PRIVATE $<$<CONFIG:Release>:/O2 /DNDEBUG>)
    target_compile_options(test_parallel PRIVATE $<$<CONFIG:Release>:/O2 /DNDEBUG>)
    target_compile_options(test_cuda PRIVATE $<$<CONFIG:Release>:/O2 /DNDEBUG>)
    target_compile_options(test_cuda_candidate PRIVATE $<$<CONFIG:Release>:/O2 /DNDEBUG>)
    target_compile_options(test_cuda_qlsa_candidate PRIVATE $<$<CONFIG:Release>:/O2 /DNDEBUG>)
    target_compile_options(test_qlsa_paper_lite PRIVATE $<$<CONFIG:Release>:/O2 /DNDEBUG>)
else()
    target_compile_options(test_small_instance PRIVATE $<$<CONFIG:Release>:-O3 -march=native -DNDEBUG>)
    target_compile_options(test_parallel PRIVATE $<$<CONFIG:Release>:-O3 -march=native -DNDEBUG>)
    target_compile_options(test_cuda PRIVATE $<$<CONFIG:Release>:-O3 -march=native -DNDEBUG>)
    target_compile_options(test_cuda_candidate PRIVATE $<$<CONFIG:Release>:-O3 -march=native -DNDEBUG>)
    target_compile_options(test_cuda_qlsa_candidate PRIVATE $<$<CONFIG:Release>:-O3 -march=native -DNDEBUG>)
    target_compile_options(test_qlsa_paper_lite PRIVATE $<$<CONFIG:Release>:-O3 -march=native -DNDEBUG>)
endif()

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
