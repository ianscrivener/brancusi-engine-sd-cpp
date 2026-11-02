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

add_executable(brancusi-engine host/main.cpp)
target_link_libraries(brancusi-engine PRIVATE bsd ${CMAKE_THREAD_LIBS_INIT})
if(APPLE AND COMMAND sd_set_macos_rpaths)
    sd_set_macos_rpaths(brancusi-engine)
endif()
install(TARGETS brancusi-engine RUNTIME)

add_executable(bsd-tests tests/test_bsd.cpp)
target_link_libraries(bsd-tests PRIVATE bsd)
