set(SIMULATOR_SRCS
    sim.hpp sim.cpp
    sim_flags.hpp sim_flags.inl
    geo_gen.hpp geo_gen.inl geo_gen.cpp
    level_gen.hpp level_gen.cpp
)

add_library(gpu_hideseek_cpu_impl STATIC
    ${SIMULATOR_SRCS}
)

target_link_libraries(gpu_hideseek_cpu_impl
    PUBLIC
        madrona_mw_core
    PRIVATE
        madrona_common
        madrona_mw_physics
        madrona_rendering_system
)

add_library(gpu_hideseek_mgr STATIC
    mgr.hpp mgr.cpp
)

target_link_libraries(gpu_hideseek_mgr
    PUBLIC
        madrona_python_utils
    PRIVATE
        gpu_hideseek_cpu_impl
        madrona_mw_cpu
        madrona_common
        madrona_importer
        madrona_physics_loader
        madrona_render
)

if (TARGET madrona_mw_gpu)
    madrona_build_compile_defns(
        OUT_TARGET
            gpu_hideseek_gpu_srcs
        SOURCES_DEFN
            GPU_HIDESEEK_SRC_LIST
        FLAGS_DEFN
            GPU_HIDESEEK_COMPILE_FLAGS 
        SRCS
            ${SIMULATOR_SRCS}
    )

    target_link_libraries(gpu_hideseek_mgr PRIVATE
        gpu_hideseek_gpu_srcs
        madrona_mw_gpu
    )
endif ()

target_compile_definitions(gpu_hideseek_mgr PRIVATE
    -DDATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../data/"
)

madrona_python_module(gpu_hideseek
    bindings.cpp
)

target_link_libraries(gpu_hideseek PRIVATE
    gpu_hideseek_mgr
)

if (TARGET madrona_viz)
    add_executable(viewer viewer.cpp)
    target_link_libraries(viewer PRIVATE
        madrona_mw_core gpu_hideseek_mgr madrona_viz)

    target_compile_definitions(viewer PRIVATE
        -DDATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../data/"
    )

    if (TARGET madrona_cuda)
        target_link_libraries(viewer PRIVATE madrona_cuda)
    endif()
endif()

add_executable(headless headless.cpp)
target_link_libraries(headless madrona_mw_core gpu_hideseek_mgr)

