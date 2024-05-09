include(ExternalProject)
add_subdirectory("${MADRONA_DIR}" madrona EXCLUDE_FROM_ALL)

set(BPS3D_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/bps3D")
set(BPS3D_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/bps3D")

ExternalProject_Add(bps3D_ext
    SOURCE_DIR "${BPS3D_SRC_DIR}"
    BINARY_DIR "${BPS3D_BUILD_DIR}"
)

add_library(bps3D INTERFACE)
target_include_directories(bps3D INTERFACE
    "${BPS3D_SRC_DIR}/include"
)
target_link_libraries(bps3D INTERFACE
    "${BPS3D_BUILD_DIR}/src/libbps3D.so"
)
add_dependencies(bps3D bps3D_ext)
