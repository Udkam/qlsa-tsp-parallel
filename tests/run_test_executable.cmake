if(NOT DEFINED TEST_STEM)
    message(FATAL_ERROR "TEST_STEM is required")
endif()

if(NOT DEFINED BINARY_DIR)
    message(FATAL_ERROR "BINARY_DIR is required")
endif()

set(candidates
    "${BINARY_DIR}/tests/${TEST_STEM}"
    "${BINARY_DIR}/tests/${TEST_STEM}.exe"
    "${BINARY_DIR}/tests/Release/${TEST_STEM}"
    "${BINARY_DIR}/tests/Release/${TEST_STEM}.exe"
    "${BINARY_DIR}/tests/Debug/${TEST_STEM}"
    "${BINARY_DIR}/tests/Debug/${TEST_STEM}.exe"
    "${BINARY_DIR}/tests/RelWithDebInfo/${TEST_STEM}"
    "${BINARY_DIR}/tests/RelWithDebInfo/${TEST_STEM}.exe"
    "${BINARY_DIR}/tests/MinSizeRel/${TEST_STEM}"
    "${BINARY_DIR}/tests/MinSizeRel/${TEST_STEM}.exe"
)

set(test_exe "")
foreach(candidate IN LISTS candidates)
    if(EXISTS "${candidate}")
        set(test_exe "${candidate}")
        break()
    endif()
endforeach()

if(test_exe STREQUAL "")
    message(FATAL_ERROR "Could not find test executable for ${TEST_STEM}")
endif()

execute_process(
    COMMAND "${test_exe}"
    RESULT_VARIABLE result
    OUTPUT_VARIABLE stdout
    ERROR_VARIABLE stderr
)

if(NOT stdout STREQUAL "")
    message(STATUS "${stdout}")
endif()

if(NOT stderr STREQUAL "")
    message(STATUS "${stderr}")
endif()

if(NOT result EQUAL 0)
    message(FATAL_ERROR "${TEST_STEM} failed with exit code ${result}")
endif()
