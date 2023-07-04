#include <madrona/viz/viewer.hpp>

#include "mgr.hpp"

#include <filesystem>

using namespace madrona;
using namespace madrona::viz;

int main(int argc, char *argv[])
{
    using namespace GPUHideSeek;

    (void)argc;
    (void)argv;

    std::array<char, 1024> import_err;
    auto render_assets = imp::ImportedAssets::importFromDisk({
        (std::filesystem::path(DATA_DIR) / "sphere.obj").c_str(),
        (std::filesystem::path(DATA_DIR) / "plane.obj").c_str(),
        (std::filesystem::path(DATA_DIR) / "cube_render.obj").c_str(),
        (std::filesystem::path(DATA_DIR) / "wall_render.obj").c_str(),
        (std::filesystem::path(DATA_DIR) / "cylinder_render.obj").c_str(),
        (std::filesystem::path(DATA_DIR) / "ramp_render.obj").c_str(),
        (std::filesystem::path(DATA_DIR) / "elongated_render.obj").c_str(),
    }, Span<char>(import_err.data(), import_err.size()));

    if (!render_assets.has_value()) {
        FATAL("Failed to load render assets: %s", import_err);
    }

    std::array<imp::SourceMaterial, 3> materials = {{
        { math::Vector4{0.4f, 0.4f, 0.4f, 0.0f} },
        { math::Vector4{1.0f, 0.1f, 0.1f, 0.0f} },
        { math::Vector4{0.1f, 0.1f, 1.0f, 0.0f} }
    }};

    const_cast<uint32_t&>(render_assets->objects[0].meshes[0].materialIDX) = 0;
    const_cast<uint32_t&>(render_assets->objects[1].meshes[0].materialIDX) = 0;
    const_cast<uint32_t&>(render_assets->objects[2].meshes[0].materialIDX) = 1;
    const_cast<uint32_t&>(render_assets->objects[3].meshes[0].materialIDX) = 0;
    const_cast<uint32_t&>(render_assets->objects[4].meshes[0].materialIDX) = 2;
    const_cast<uint32_t&>(render_assets->objects[5].meshes[0].materialIDX) = 0;
    const_cast<uint32_t&>(render_assets->objects[6].meshes[0].materialIDX) = 0;

    uint32_t num_worlds = 1;

    Viewer viewer({
        .gpuID = 0,
        .renderWidth = 2730,
        .renderHeight = 1536,
        .numWorlds = num_worlds,
        .maxViewsPerWorld = 6,
        .maxInstancesPerWorld = 1000,
        .execMode = ExecMode::CPU,
    });

    viewer.loadObjects(render_assets->objects, materials);

    viewer.configureLighting({
        { true, math::Vector3{1.0f, 1.0f, -1.5f}, math::Vector3{1.0f, 1.0f, 1.0f} }
    });

    Manager mgr({
        .execMode = ExecMode::CPU,
        .gpuID = 0,
        .numWorlds = num_worlds,
        .renderWidth = 0,
        .renderHeight = 0,
        .autoReset = false,
        .enableBatchRender = false,
        .debugCompile = false,
    }, viewer.rendererBridge());

    for (uint32_t i = 0; i < num_worlds; i++) {
        mgr.triggerReset(i, 1, 2, 2);
    }

    viewer.loop([&mgr](CountT world_idx, CountT agent_idx,
                       const Viewer::UserInput &input) {
        using Key = Viewer::KeyboardKey;

        int32_t x = 0;
        int32_t y = 0;
        int32_t r = 0;
        bool g = false;
        bool l = false;

        if (input.keyPressed(Key::R)) {
            mgr.triggerReset(world_idx, 1, 2, 2);
        }

        if (input.keyPressed(Key::K1)) {
            mgr.triggerReset(world_idx, 1, 2, 2);
        }

        if (input.keyPressed(Key::K2)) {
            mgr.triggerReset(world_idx, 2, 2, 2);
        }

        if (input.keyPressed(Key::K3)) {
            mgr.triggerReset(world_idx, 3, 2, 2);
        }

        if (input.keyPressed(Key::K4)) {
            mgr.triggerReset(world_idx, 4, 2, 2);
        }

        if (input.keyPressed(Key::K5)) {
            mgr.triggerReset(world_idx, 5, 2, 2);
        }

        if (input.keyPressed(Key::K6)) {
            mgr.triggerReset(world_idx, 6, 2, 2);
        }

        if (input.keyPressed(Key::K7)) {
            mgr.triggerReset(world_idx, 7, 2, 2);
        }

        if (input.keyPressed(Key::K8)) {
            mgr.triggerReset(world_idx, 8, 2, 2);
        }

        if (input.keyPressed(Key::K9)) {
            mgr.triggerReset(world_idx, 9, 2, 2);
        }

        if (input.keyPressed(Key::W)) {
            y += 5;
        }
        if (input.keyPressed(Key::S)) {
            y -= 5;
        }

        if (input.keyPressed(Key::D)) {
            x += 5;
        }
        if (input.keyPressed(Key::A)) {
            x -= 5;
        }

        if (input.keyPressed(Key::Q)) {
            r += 5;
        }
        if (input.keyPressed(Key::E)) {
            r -= 5;
        }

        if (input.keyPressed(Key::G)) {
            g = true;
        }
        if (input.keyPressed(Key::L)) {
            l = true;
        }

        mgr.setAction(world_idx * 4 + agent_idx, x, y, r, g, l);
    }, [&mgr]() {
        mgr.step();
    });
}
