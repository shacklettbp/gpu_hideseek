#include <madrona/viz/viewer.hpp>

#include "mgr.hpp"

#include <filesystem>

using namespace madrona;
using namespace madrona::viz;

static inline float srgbToLinear(float srgb)
{
    if (srgb <= 0.04045f) {
        return srgb / 12.92f;
    }

    return powf((srgb + 0.055f) / 1.055f, 2.4f);
}

static inline math::Vector4 rgb8ToFloat(uint8_t r, uint8_t g, uint8_t b)
{
    return {
        srgbToLinear((float)r / 255.f),
        srgbToLinear((float)g / 255.f),
        srgbToLinear((float)b / 255.f),
        1.f,
    };
}

int main(int argc, char *argv[])
{
    using namespace GPUHideSeek;

    (void)argc;
    (void)argv;

    std::array<char, 1024> import_err;
    auto render_assets = imp::ImportedAssets::importFromDisk({
        (std::filesystem::path(DATA_DIR) / "sphere.obj").string().c_str(),
        (std::filesystem::path(DATA_DIR) / "plane.obj").string().c_str(),
        (std::filesystem::path(DATA_DIR) / "cube_render.obj").string().c_str(),
        (std::filesystem::path(DATA_DIR) / "wall_render.obj").string().c_str(),
        (std::filesystem::path(DATA_DIR) / "agent_render.obj").string().c_str(),
        (std::filesystem::path(DATA_DIR) / "ramp_render.obj").string().c_str(),
        (std::filesystem::path(DATA_DIR) / "elongated_render.obj").string().c_str(),
    }, Span<char>(import_err.data(), import_err.size()));

    if (!render_assets.has_value()) {
        FATAL("Failed to load render assets: %s", import_err.data());
    }

    auto materials = std::to_array<imp::SourceMaterial>({
        { math::Vector4{0.4f, 0.4f, 0.4f, 0.0f}, -1, 0.8f, 0.2f,},
        { math::Vector4{1.0f, 0.1f, 0.1f, 0.0f}, -1, 0.8f, 0.2f,},
        { math::Vector4{1.f, 1.f, 1.f, 0.0f}, 1, 0.5f, 1.0f,},
        { math::Vector4{0.5f, 0.3f, 0.3f, 0.0f},  0, 0.8f, 0.2f,},
        { rgb8ToFloat(191, 108, 10), -1, 0.8f, 0.2f },
        { rgb8ToFloat(12, 144, 150), -1, 0.8f, 0.2f },
        { rgb8ToFloat(230, 230, 230),   -1, 0.8f, 1.0f },
    });

    const_cast<uint32_t&>(render_assets->objects[0].meshes[0].materialIDX) = 0;
    const_cast<uint32_t&>(render_assets->objects[1].meshes[0].materialIDX) = 3;
    const_cast<uint32_t&>(render_assets->objects[2].meshes[0].materialIDX) = 1;
    const_cast<uint32_t&>(render_assets->objects[3].meshes[0].materialIDX) = 0;
    const_cast<uint32_t&>(render_assets->objects[4].meshes[0].materialIDX) = 2;
    const_cast<uint32_t&>(render_assets->objects[4].meshes[1].materialIDX) = 6;
    const_cast<uint32_t&>(render_assets->objects[4].meshes[2].materialIDX) = 6;
    const_cast<uint32_t&>(render_assets->objects[5].meshes[0].materialIDX) = 4;
    const_cast<uint32_t&>(render_assets->objects[6].meshes[0].materialIDX) = 5;

    uint32_t num_worlds = 1;

    Viewer viewer({
        .gpuID = 0,
        .renderWidth = 1920,
        .renderHeight = 1080,
        .numWorlds = num_worlds,
        .maxViewsPerWorld = 6,
        .maxInstancesPerWorld = 1000,
        .defaultSimTickRate = 30,
        .execMode = ExecMode::CPU,
    });

    viewer.loadObjects(render_assets->objects, materials, {
        { (std::filesystem::path(DATA_DIR) / "green_grid.png").string().c_str() },
        { (std::filesystem::path(DATA_DIR) / "smile.png").string().c_str() },
    });

    viewer.configureLighting({
        { true, math::Vector3{1.0f, 1.0f, -2.f}, math::Vector3{1.0f, 1.0f, 1.0f} }
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

    uint32_t steps_per_episode = 480;
    uint32_t cur_episode_step = 0;
    viewer.loop([&mgr](CountT world_idx, CountT agent_idx,
                       const Viewer::UserInput &input) {
        using Key = Viewer::KeyboardKey;

        int32_t x = 0;
        int32_t y = 0;
        int32_t r = 0;
        bool g = false;
        bool l = false;

        if (getenv("FAKE_INPUT")) {
            return;
        }

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
    }, [&mgr, &cur_episode_step, steps_per_episode]() {
        cur_episode_step++;

        if (getenv("FAKE_INPUT")) {
            if (cur_episode_step == steps_per_episode) {
                cur_episode_step = 0;
                mgr.triggerReset(0, 1, 2, 2);
            }

            int32_t x = 0;
            int32_t y = 0;
            int32_t r = 0;
            bool g = false;
            bool l = false;

            y = 5;

            for (int i = 0; i < 4; i++) {
                mgr.setAction(i, x, y, r, g, l);
            } 
        }

        mgr.step();
    });
}
