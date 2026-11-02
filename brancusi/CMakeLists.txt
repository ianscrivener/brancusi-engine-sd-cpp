# Brancusi additions to stable-diffusion.cpp.
#   bsd              static library: the bsd_* C API (Layer 1 and Layer 2)
#   bsd_l2           object library: Layer 2 sources (upstream internals)
#   brancusi-engine  host executable: JSON lines on stdin and stdout
#   bsd-tests        unit tests (no model needed)

# Layer 2 sources use upstream internal headers. Build them with the same include folders and
# definitions as the upstream library, so the internal structs have the same layout.
add_library(bsd_l2 OBJECT
    src/l2_inspect.cpp
)
target_include_directories(bsd_l2 PRIVATE include $<TARGET_PROPERTY:stable-diffusion,INCLUDE_DIRECTORIES>)
target_compile_definitions(bsd_l2 PRIVATE $<TARGET_PROPERTY:stable-diffusion,COMPILE_DEFINITIONS>)
target_link_libraries(bsd_l2 PRIVATE ggml)
target_compile_features(bsd_l2 PRIVATE cxx_std_17)

add_library(bsd STATIC
    src/l1_engine.cpp
    $<TARGET_OBJECTS:bsd_l2>
)
target_include_directories(bsd PUBLIC include)
target_link_libraries(bsd PUBLIC stable-diffusion)
target_compile_features(bsd PUBLIC cxx_std_17)

# Engine release version. Every push bumps the patch number. Release tags are "brancusi-v<version>".
set(BRANCUSI_VERSION 0.1.1)
# This fork starts from this upstream stable-diffusion.cpp commit. Update it after each upstream merge
# (git merge-base HEAD upstream/master).
set(BRANCUSI_SD_BASE 37bc88b)

# Release build (HEAD is at the clean tag brancusi-v<version>): "0.1.1+sd.37bc88b".
# Other builds: "0.1.1-dev.<commits since the last tag>.g<commit>[.dirty]+sd.37bc88b".
set(BRANCUSI_ENGINE_VERSION "${BRANCUSI_VERSION}-dev+sd.${BRANCUSI_SD_BASE}")
if(GIT_EXE)
    execute_process(COMMAND ${GIT_EXE} describe --tags --match "brancusi-v*" --long --abbrev=7 --dirty=-dirty
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        OUTPUT_VARIABLE BRANCUSI_DESCRIBE
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(BRANCUSI_DESCRIBE MATCHES "^brancusi-v(.+)-([0-9]+)-g([0-9a-f]+)(-dirty)?$")
        set(_tag "${CMAKE_MATCH_1}")
        set(_count "${CMAKE_MATCH_2}")
        set(_hash "${CMAKE_MATCH_3}")
        set(_dirty "${CMAKE_MATCH_4}")
        if(_tag STREQUAL BRANCUSI_VERSION AND _count STREQUAL "0" AND NOT _dirty)
            set(BRANCUSI_ENGINE_VERSION "${BRANCUSI_VERSION}+sd.${BRANCUSI_SD_BASE}")
        else()
            set(_dev "${BRANCUSI_VERSION}-dev.${_count}.g${_hash}")
            if(_dirty)
                set(_dev "${_dev}.dirty")
            endif()
            set(BRANCUSI_ENGINE_VERSION "${_dev}+sd.${BRANCUSI_SD_BASE}")
        endif()
    endif()
endif()
message(STATUS "brancusi-engine version ${BRANCUSI_ENGINE_VERSION}")

add_executable(brancusi-engine host/main.cpp)
target_compile_definitions(brancusi-engine PRIVATE BRANCUSI_ENGINE_VERSION="${BRANCUSI_ENGINE_VERSION}")
target_link_libraries(brancusi-engine PRIVATE bsd ${CMAKE_THREAD_LIBS_INIT})
if(APPLE AND COMMAND sd_set_macos_rpaths)
    sd_set_macos_rpaths(brancusi-engine)
endif()
install(TARGETS brancusi-engine RUNTIME)

add_executable(bsd-tests tests/test_bsd.cpp)
target_link_libraries(bsd-tests PRIVATE bsd)
