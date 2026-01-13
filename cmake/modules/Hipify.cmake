# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

find_package(Python3 COMPONENTS Interpreter REQUIRED)

function(auto_set_source_files_hip_language)
  foreach(f ${ARGN})
    if(f MATCHES ".*\\.(cu|cuh)$")
      set_source_files_properties(${f} PROPERTIES LANGUAGE HIP)
    endif()
  endforeach()
endfunction()

# cuda_dir is an absolute path or path relative to PROJECT_SOURCE_DIR
function(hipify cuda_dir in_excluded_file_patterns out_generated_cc_files out_generated_cu_files)
  # Get hipify tool path from environment variable
  if(DEFINED ENV{HIPIFY_TOOL_PATH})
    set(hipify_tool $ENV{HIPIFY_TOOL_PATH})
  else()
    message(FATAL_ERROR "HIPIFY_TOOL_PATH environment variable is not set. Please set it to the path of your hipify tool.")
  endif()

  # Verify the tool exists
  if(NOT EXISTS ${hipify_tool})
    message(FATAL_ERROR "Hipify tool not found at: ${hipify_tool}")
  endif()

  message(STATUS "Using hipify tool: ${hipify_tool}")

  # Determine the search directory
  if(IS_ABSOLUTE ${cuda_dir})
    set(search_dir ${cuda_dir})
  else()
    # Assume relative to PROJECT_SOURCE_DIR if not absolute
    set(search_dir ${PROJECT_SOURCE_DIR}/${cuda_dir})
  endif()

  # Verify the directory exists
  if(NOT EXISTS ${search_dir})
    message(FATAL_ERROR "Directory to hipify does not exist: ${search_dir}")
  endif()

  # Find files in the specified directory
  file(GLOB_RECURSE srcs CONFIGURE_DEPENDS
    "${search_dir}/*.cc"
    "${search_dir}/*.cpp"
    "${search_dir}/*.cu"
    "${search_dir}/*.cuh"
    "${search_dir}/*.h"
    "${search_dir}/*.hpp"
  )

  # do exclusion
  set(excluded_file_patterns ${${in_excluded_file_patterns}})

  # Filter out excluded patterns
  set(filtered_srcs)
  foreach(src ${srcs})
    set(should_exclude FALSE)
    foreach(pattern ${excluded_file_patterns})
      if(src MATCHES "${pattern}")
        set(should_exclude TRUE)
        message(STATUS "Excluding file from hipification: ${src}")
        break()
      endif()
    endforeach()
    
    if(NOT should_exclude)
      list(APPEND filtered_srcs ${src})
    endif()
  endforeach()

  # Use filtered list instead of original
  set(srcs ${filtered_srcs})

  # Get actual number of files found
  list(LENGTH srcs num_files)
  message(STATUS "Hipifying directory: ${search_dir}")
  message(STATUS "Found ${num_files} files to process")

  # Extract a clean name for the target from the directory path
  string(REPLACE "/" "_" target_name ${cuda_dir})
  string(REPLACE "." "_" target_name ${target_name})

  # Check if HIPIFY_PERL_PATH environment variable is set and points to a valid file
  if(DEFINED ENV{HIPIFY_PERL_PATH})
    set(HIPIFY_PERL_PATH $ENV{HIPIFY_PERL_PATH})
  else()
    message(FATAL_ERROR "HIPIFY_PERL_PATH environment variable is not set. Please set it to the path of your hipify_perl script.")
  endif()

  if(NOT EXISTS ${HIPIFY_PERL_PATH})
    message(FATAL_ERROR "HIPIFY_PERL_PATH script not found at: ${HIPIFY_PERL_PATH}")
  endif()

  # Debug: Print Python interpreter and script paths
  message(STATUS "Python interpreter: ${Python3_EXECUTABLE}")
  message(STATUS "Hipify script: ${hipify_tool}")
  message(STATUS "Hipify perl: ${HIPIFY_PERL_PATH}")

  # Test if we can run the hipify tool at all
  execute_process(
    COMMAND ${Python3_EXECUTABLE} ${hipify_tool} --help
    RESULT_VARIABLE help_result
    OUTPUT_VARIABLE help_output
    ERROR_VARIABLE help_error
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_STRIP_TRAILING_WHITESPACE
  )
  
  if(NOT help_result EQUAL 0)
    message(WARNING "Cannot run hipify tool --help:")
    message(WARNING "  Exit code: ${help_result}")
    message(WARNING "  Output: ${help_output}")
    message(WARNING "  Error: ${help_error}")
  endif()

  set(successful_count 0)
  set(failed_count 0)

  foreach(f ${srcs})
    message(STATUS "Hipifying: ${f}")
    # Generate output file name by replacing 'cuda' with 'rocm' in the file name
    get_filename_component(f_dir "${f}" DIRECTORY)
    get_filename_component(f_name "${f}" NAME)
    string(REPLACE "cuda" "rocm" out_f_name "${f_name}")
    string(REPLACE "curand" "rocrand" out_f_name "${out_f_name}")
    set(out_f "${f_dir}/${out_f_name}")
    
    # Build the command as a list
    set(hipify_cmd 
      ${Python3_EXECUTABLE} 
      ${hipify_tool}
      --hipify_perl
      ${HIPIFY_PERL_PATH}
      ${f}
      -o
      ${out_f}
    )
    
    # Debug: Print the exact command being run
    string(REPLACE ";" " " hipify_cmd_string "${hipify_cmd}")
    message(STATUS "  Running: ${hipify_cmd_string}")
    
    if(EXISTS "${f}.hipified")
      message(STATUS "  File already hipified: ${f}")
    else()
      # Run hipification immediately during configuration
      execute_process(
        COMMAND ${hipify_cmd}
        RESULT_VARIABLE hipify_result
        OUTPUT_VARIABLE hipify_output
        ERROR_VARIABLE hipify_error
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_STRIP_TRAILING_WHITESPACE
      )
      
      if(NOT hipify_result EQUAL 0)
        message(WARNING "Hipification failed for ${f}:")
        message(WARNING "  Exit code: ${hipify_result}")
        message(WARNING "  Error: ${hipify_error}")
        message(WARNING "  Output: ${hipify_output}")
        math(EXPR failed_count "${failed_count} + 1")
      else()
        message(STATUS "  Successfully hipified: ${f}")
        if(hipify_output)
          message(STATUS "  Output: ${hipify_output}")
        endif()
        # Create marker file to indicate successful hipification
        file(TOUCH "${f}.hipified")
        math(EXPR successful_count "${successful_count} + 1")
      endif()
    endif()

    # Create a dummy target to track hipification
    list(APPEND hipified_markers ${f}.hipified)

    if(f MATCHES ".*\\.(cu|cuh)$")
      list(APPEND generated_cu_files ${out_f})
    else()
      list(APPEND generated_cc_files ${out_f})
    endif()

    # If input and output files are different, remove the oirignal file to avoid stale results
    if(NOT f STREQUAL out_f AND EXISTS ${out_f})
      file(REMOVE ${f})
    endif()
  endforeach()

  # Create a custom target to ensure all files are hipified
  add_custom_target(hipify_${target_name}_files ALL DEPENDS ${hipified_markers})

  auto_set_source_files_hip_language(${generated_cu_files})
  set(${out_generated_cc_files} ${generated_cc_files} PARENT_SCOPE)
  set(${out_generated_cu_files} ${generated_cu_files} PARENT_SCOPE)
  
  # Get actual counts
  list(LENGTH generated_cu_files num_cu_files)
  list(LENGTH generated_cc_files num_cc_files)
  
  message(STATUS "Hipification complete for ${search_dir}")
  message(STATUS "  Total files processed: ${num_files}")
  message(STATUS "  Successful: ${successful_count}")
  message(STATUS "  Failed: ${failed_count}")
  message(STATUS "  CU files: ${num_cu_files}")
  message(STATUS "  CC/H files: ${num_cc_files}")
endfunction()
