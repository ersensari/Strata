if(NOT STRATA_ENABLE_CUDA OR NOT STRATA_NATIVE_EXPERTS)
  message(FATAL_ERROR "STRATA_ORNITH_NATIVE has only been implemented for CUDA with native experts")
endif()
target_compile_definitions(strata_kernels PRIVATE STRATA_ORNITH_NATIVE=1)
target_compile_definitions(strata_kernels_cpu PUBLIC STRATA_ORNITH_NATIVE=1)
# Same independent ggml float/CPU/GPU reference check at Ornith's actual expert dimensions.
add_executable(ornith_native_parity src/kernels/native_expert_parity.cpp)
target_compile_definitions(ornith_native_parity PRIVATE STRATA_NATIVE_TEST_H=3072 STRATA_NATIVE_TEST_FF=1024)
target_include_directories(ornith_native_parity PRIVATE ${_strata_gpu_include_directories})
target_link_libraries(ornith_native_parity PRIVATE strata_engine strata_kernels strata_kernels_cpu strata_artifact
                                                 ${_strata_gpu_runtime_target})
