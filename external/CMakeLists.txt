include(ExternalProject)
add_subdirectory("${MADRONA_DIR}" madrona EXCLUDE_FROM_ALL)

set(BPS3D_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/bps3D")

get_target_property(LIBCXX_INC_DIR madrona_libcxx_hdrs
    INTERFACE_SYSTEM_INCLUDE_DIRECTORIES)

string(REGEX REPLACE "^\\$<BUILD_INTERFACE:" "" LIBCXX_INC_DIR ${LIBCXX_INC_DIR})
string(REGEX REPLACE ">$" "" LIBCXX_INC_DIR ${LIBCXX_INC_DIR})
message(STATUS ${LIBCXX_INC_DIR})

get_target_property(LIBCXX_LIB madrona_libcxx
    INTERFACE_LINK_LIBRARIES)

set(BPS3D_LIBCXX_INC_FLAGS "-nostdinc++ -isystem ${LIBCXX_INC_DIR}")
set(BPS3D_LIBCXX_LINK_FLAGS "-nostdlib++ ${LIBCXX_LIB}")
    

ExternalProject_Add(bps3D_ext
    SOURCE_DIR "${BPS3D_SRC_DIR}"
    INSTALL_DIR "${CMAKE_BINARY_DIR}"
    CMAKE_ARGS 
        "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
        "-DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}"
        "-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}"
        "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
        "-DCMAKE_CXX_FLAGS=${BPS3D_LIBCXX_INC_FLAGS} ${CMAKE_CXX_FLAGS}"
        "-DCMAKE_EXE_LINKER_FLAGS=${BPS3D_LIBCXX_LINK_FLAGS} ${CMAKE_EXE_LINKER_FLAGS}"
        "-DCMAKE_SHARED_LINKER_FLAGS=${BPS3D_LIBCXX_LINK_FLAGS} ${CMAKE_SHARED_LINKER_FLAGS}"
        "-DCMAKE_MODULE_LINKER_FLAGS=${BPS3D_LIBCXX_LINK_FLAGS} ${CMAKE_MODULE_LINKER_FLAGS}"
    STEP_TARGETS install
)

add_library(bps3D_imp SHARED IMPORTED)
target_include_directories(bps3D_imp INTERFACE
    "${BPS3D_SRC_DIR}/include"
)
set_target_properties(bps3D_imp PROPERTIES
    IMPORTED_LOCATION "${CMAKE_BINARY_DIR}/libbps3D.so"
)

add_library(bps3D INTERFACE)
add_dependencies(bps3D bps3D_ext_install)
target_link_libraries(bps3D INTERFACE bps3D_imp)

add_subdirectory(bps3D/external/glm)
