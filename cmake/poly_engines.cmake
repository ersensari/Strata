# PolyStrata engine integration, imported from ersensari/PolyStrata f6c93f63.
if(NOT STRATA_NATIVE_EXPERTS OR NOT (STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP) OR STRATA_HIP_GFX906)
  message(FATAL_ERROR "STRATA_POLY_ENGINES requires native experts and a CUDA or wave32 HIP build")
endif()
add_library(strata_poly_native STATIC src/poly/native_mmvq.cu src/poly/iq_kernels.cu)
target_include_directories(strata_poly_native PRIVATE include src/kernels/cuda third_party/ggml ${_strata_gpu_include_directories})
target_compile_options(strata_poly_native PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
target_link_libraries(strata_poly_native PUBLIC ${_strata_gpu_runtime_target})
# ---- strata-poly: GLM 5.3 Flash (llama.cpp's glm5-next), read from its GGUF ----
#
# A second engine beside `strata`.  It shares the quantized matvec kernels and the GGUF reader, and takes the
# CPU experts' dot products from ggml-cpu.  On HIP it is the same CUDA-shaped code through the hip_compat headers
# (the wave32 RDNA backend; gfx906's wave64 compat layer is not supported here).
if(STRATA_NATIVE_EXPERTS AND (STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP) AND NOT STRATA_HIP_GFX906)
  # The prompt path's quantized products: llama.cpp's MMQ kernels (ggml-cuda, MIT) for the types the model's
  # GGUFs keep their matrices in (the list in src/poly/mmq.cu), from the checkout strata_mmq is compiled from and
  # with the same host symbols.  A library of its own, so that what `strata` multiplies with MMQ stays as it is.
  if(NOT _strata_mmq)   # a HIP build without STRATA_PREFILL_MMQ has not set these up yet
    if(STRATA_GGML_DIR)
      set(_strata_gg ${STRATA_GGML_DIR}/ggml)
    else()
      set(_strata_gg ${strata_llamacpp_SOURCE_DIR}/ggml)
    endif()
    set(_strata_mmq ${_strata_gg}/src/ggml-cuda)
  endif()
  set(_strata_poly_mmq_instances "")
  foreach(_t q4_0 q4_1 q5_0 q5_1 q8_0 q2_k q3_k q4_k q5_k q6_k iq1_s iq2_xxs iq2_xs iq2_s iq3_xxs iq3_s iq4_nl
             iq4_xs)
    list(APPEND _strata_poly_mmq_instances ${_strata_mmq}/template-instances/mmq-instance-${_t}.cu)
  endforeach()
  # MXFP4: on RTX 50 llama.cpp's tiles use the FP4 MMA with block scales, which exists only in the arch-specific
  # target, so on CUDA its instance alone is built with 12x -> 12xa, as Strata's W4A4 unit is.  A Blackwell other
  # than 12.0 has no image of it, and Mmq::fits() leaves MXFP4 out there.
  set(_strata_poly_mxfp4 ${_strata_mmq}/template-instances/mmq-instance-mxfp4.cu)
  if(STRATA_ENABLE_HIP)
    list(APPEND _strata_poly_mmq_instances ${_strata_poly_mxfp4})
  else()
    set(_strata_poly_fp4_archs ${CMAKE_CUDA_ARCHITECTURES})
    list(TRANSFORM _strata_poly_fp4_archs REPLACE "^(12[0-9])$" "\\1a")
    list(TRANSFORM _strata_poly_fp4_archs REPLACE "^(12[0-9])-(real|virtual)$" "\\1a-\\2")
    add_library(strata_poly_mmq_mxfp4 OBJECT ${_strata_poly_mxfp4})
    set_target_properties(strata_poly_mmq_mxfp4 PROPERTIES CUDA_ARCHITECTURES "${_strata_poly_fp4_archs}")
    target_include_directories(strata_poly_mmq_mxfp4 PRIVATE ${_strata_gg}/include ${_strata_gg}/src ${_strata_mmq})
    target_compile_options(strata_poly_mmq_mxfp4 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda -use_fast_math>)
    if(MSVC)
      target_compile_options(strata_poly_mmq_mxfp4 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/bigobj>)
    endif()
    list(APPEND _strata_poly_mmq_instances $<TARGET_OBJECTS:strata_poly_mmq_mxfp4>)
  endif()
  if(STRATA_ENABLE_HIP)
    # CMake does not infer HIP from the .cu suffixes outside src/ (and drops a .cu source it has no language
    # for), and ggml's MMQ sources take its native HIP path (vendors/hip.h) instead of the hip_compat headers,
    # as strata_mmq's does.
    set_source_files_properties(${_strata_mmq}/quantize.cu ${_strata_poly_mmq_instances} PROPERTIES LANGUAGE HIP)
  endif()
  add_library(strata_poly_mmq STATIC src/poly/mmq.cu src/poly/fattn.cu src/prefill/ggml_cuda_host.cu ${_strata_mmq}/quantize.cu
              ${_strata_poly_mmq_instances})
  target_include_directories(strata_poly_mmq PRIVATE ${_strata_gg}/include ${_strata_gg}/src ${_strata_mmq}
                                                    ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_compile_options(strata_poly_mmq PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda -use_fast_math>)
  if(MSVC)
    target_compile_options(strata_poly_mmq PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/bigobj>)
  endif()
  if(STRATA_ENABLE_HIP)
    target_compile_definitions(strata_poly_mmq PRIVATE GGML_USE_HIP=1)
    target_link_libraries(strata_poly_mmq PUBLIC ggml-base hip::host)
  else()
    target_link_libraries(strata_poly_mmq PUBLIC ggml-base ${_strata_gpu_runtime_target})
  endif()

  add_executable(strata-poly src/poly/main.cpp src/poly/engine.cpp src/poly/glm5next.cpp src/poly/qwen35.cpp
                            src/poly/gemma4.cpp src/poly/deepseek4.cpp src/poly/lfm2moe.cpp src/poly/mellum.cpp src/poly/cpu_experts.cpp
                            src/poly/kernels.cu src/poly/kernels_dsv4.cu src/core/conversation_memory.cpp)
  target_include_directories(strata-poly PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${_strata_gpu_include_directories})
  # the driver's virtual memory calls: the attention caches grow with the conversation; on HIP they are HIP's
  # memory management calls, which the HIP runtime itself provides (hip_compat/cuda.h)
  if(STRATA_ENABLE_HIP)
    set(_strata_poly_driver "")
  else()
    set(_strata_poly_driver CUDA::cuda_driver)
  endif()
  target_link_libraries(strata-poly PRIVATE strata_poly_native strata_poly_mmq strata_kernels strata_spec ggml-cpu ggml-base
                                           ${_strata_gpu_blas_target} ${_strata_gpu_runtime_target}
                                           ${_strata_poly_driver}
                                           Threads::Threads strata_kernels_cpu)
  # needs the model file, so it is run by hand: poly_expert_parity MODEL.gguf
  add_executable(poly_expert_parity src/poly/expert_parity.cpp src/poly/cpu_experts.cpp src/poly/kernels.cu)
  target_include_directories(poly_expert_parity PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
                                                       ${_strata_gpu_include_directories})
  target_link_libraries(poly_expert_parity PRIVATE strata_poly_native strata_poly_mmq strata_kernels ggml-cpu ggml-base
                                                  ${_strata_gpu_blas_target} ${_strata_gpu_runtime_target}
                                                  Threads::Threads strata_kernels_cpu)
  # a read with the draft block repeats bit for bit; needs the model file, so it is run by hand:
  # poly_draft_race_test MODEL.gguf [RUNS] [LAYER_SPLIT]
  add_executable(poly_draft_race_test src/poly/draft_race_test.cpp src/poly/engine.cpp src/poly/glm5next.cpp
                                      src/poly/qwen35.cpp src/poly/gemma4.cpp src/poly/deepseek4.cpp
                                      src/poly/lfm2moe.cpp src/poly/mellum.cpp src/poly/cpu_experts.cpp src/poly/kernels.cu src/poly/kernels_dsv4.cu
                                      src/core/conversation_memory.cpp)
  target_include_directories(poly_draft_race_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
                                                         ${_strata_gpu_include_directories})
  target_link_libraries(poly_draft_race_test PRIVATE strata_poly_native strata_poly_mmq strata_kernels strata_spec ggml-cpu ggml-base
                                                    ${_strata_gpu_blas_target} ${_strata_gpu_runtime_target}
                                                    ${_strata_poly_driver} Threads::Threads strata_kernels_cpu)
  # a prompt batch's routing on the card against the hosts' routing; needs a card, so it is run by hand
  add_executable(poly_route_test src/poly/route_test.cu src/poly/kernels.cu)
  target_include_directories(poly_route_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
                                                    ${_strata_gpu_include_directories})
  target_link_libraries(poly_route_test PRIVATE ${_strata_gpu_blas_target} ${_strata_gpu_runtime_target})
  # the fused residual sum, RMS norm and Q8_1 quantization against the three launches; needs a card, run by hand
  add_executable(poly_norm_q8_test src/poly/norm_q8_test.cu src/poly/kernels.cu)
  target_include_directories(poly_norm_q8_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
                                                      ${_strata_gpu_include_directories})
  target_link_libraries(poly_norm_q8_test PRIVATE strata_poly_native strata_kernels ${_strata_gpu_blas_target}
                                                  ${_strata_gpu_runtime_target})
  # a prompt batch's products with few outputs against the same sums on the host; needs a card, run by hand
  add_executable(poly_gemm_small_test src/poly/gemm_small_test.cu src/poly/kernels.cu)
  target_include_directories(poly_gemm_small_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
                                                         ${_strata_gpu_include_directories})
  target_link_libraries(poly_gemm_small_test PRIVATE ${_strata_gpu_blas_target} ${_strata_gpu_runtime_target})
  # attend()'s blocks of rows against its other kernels and itself; needs a card, so it is run by hand
  add_executable(poly_attend_test src/poly/attend_test.cu src/poly/kernels.cu)
  target_include_directories(poly_attend_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
                                                     ${_strata_gpu_include_directories})
  target_link_libraries(poly_attend_test PRIVATE ${_strata_gpu_blas_target} ${_strata_gpu_runtime_target})
  # fattn_q8() and fattn_f16() (llama.cpp's flash-attention kernel) against attend_batch(), and with `dec` fdec_q8()
  # against attend(); needs a card, run by hand
  add_executable(poly_fattn_test src/poly/fattn_test.cu src/poly/kernels.cu)
  target_include_directories(poly_fattn_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/src/poly
                                                    ${_strata_gpu_include_directories})
  target_link_libraries(poly_fattn_test PRIVATE strata_poly_mmq ${_strata_gpu_blas_target} ${_strata_gpu_runtime_target})
  # LFM2's short convolution against the same arithmetic on the host; needs a card, so it is run by hand
  add_executable(poly_shortconv_test src/poly/shortconv_test.cu src/poly/kernels.cu)
  target_include_directories(poly_shortconv_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
                                                        ${_strata_gpu_include_directories})
  target_link_libraries(poly_shortconv_test PRIVATE ${_strata_gpu_blas_target} ${_strata_gpu_runtime_target})
  # DeepSeek V4's kernels against the same arithmetic on the host; needs a card, so it is run by hand
  add_executable(poly_dsv4_test src/poly/dsv4_test.cu src/poly/kernels_dsv4.cu)
  target_include_directories(poly_dsv4_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
                                                   ${_strata_gpu_include_directories})
  target_link_libraries(poly_dsv4_test PRIVATE ${_strata_gpu_runtime_target})
  add_executable(poly_cpu_iq_test src/poly/cpu_iq_test.cpp src/poly/cpu_experts.cpp)
  target_include_directories(poly_cpu_iq_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${_strata_gpu_include_directories})
  target_link_libraries(poly_cpu_iq_test PRIVATE strata_artifact strata_kernels_cpu ggml-cpu ggml-base Threads::Threads)
  # the pool of the CPU experts, without a model
  add_executable(poly_workers_test src/poly/workers_test.cpp src/poly/cpu_experts.cpp)
  target_include_directories(poly_workers_test PRIVATE ${_strata_gg}/include ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(poly_workers_test PRIVATE ggml-cpu ggml-base Threads::Threads strata_kernels_cpu)
  add_test(NAME poly_workers_test COMMAND poly_workers_test)
  # the card's MXFP4 product against ggml's dequantizer on random blocks, without a model; needs a card, run by hand
  add_executable(poly_mxfp4_test src/poly/mxfp4_test.cu)
  target_include_directories(poly_mxfp4_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${_strata_gg}/include
                                                    ${_strata_gpu_include_directories})
  target_link_libraries(poly_mxfp4_test PRIVATE strata_poly_native strata_kernels ggml-base ${_strata_gpu_runtime_target})
  if(STRATA_BUILD_TESTS)
    add_executable(poly_mellum_math_test src/poly/mellum_math_test.cpp)
    target_include_directories(poly_mellum_math_test PRIVATE ${_strata_gg}/include)
    target_link_libraries(poly_mellum_math_test PRIVATE ggml-cpu ggml-base Threads::Threads)
    add_test(NAME poly_mellum_math_test COMMAND poly_mellum_math_test)
  endif()
  # the Q8_0 KV cache kernels (store_q8 / store_q8_at) against a host reference (GPU, synthetic, no model)
  if(STRATA_BUILD_TESTS)
    add_executable(poly_kv_q8_test src/poly/kv_q8_test.cpp src/poly/kernels.cu)
    target_include_directories(poly_kv_q8_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
                                                       ${_strata_gpu_include_directories})
    target_link_libraries(poly_kv_q8_test PRIVATE ${_strata_gpu_blas_target} ${_strata_gpu_runtime_target}
                                                  Threads::Threads)
    add_test(NAME poly_kv_q8_test COMMAND poly_kv_q8_test)
  endif()
endif()
