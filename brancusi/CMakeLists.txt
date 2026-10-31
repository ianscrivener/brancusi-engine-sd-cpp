# Brancusi additions to stable-diffusion.cpp.
#   bsd              static library: the bsd_* C API (Layer 1 and Layer 2)
#   brancusi-engine  host executable: JSON lines on stdin and stdout
#   bsd-tests        unit tests (no model needed)

add_library(bsd STATIC
    src/l1_engine.cpp
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
