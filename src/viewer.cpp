#include <madrona/viz/viewer.hpp>

#include "mgr.hpp"

#include <filesystem>
#include <fstream>

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

static HeapArray<int32_t> readReplayLog(const char *path)
{
    std::ifstream replay_log(path, std::ios::binary);
    replay_log.seekg(0, std::ios::end);
    int64_t size = replay_log.tellg();
    replay_log.seekg(0, std::ios::beg);

    HeapArray<int32_t> log(size / sizeof(int32_t));

    replay_log.read((char *)log.data(), (size / sizeof(int32_t)) * sizeof(int32_t));

    return log;
}

int main(int argc, char *argv[])
{
    using namespace GPUHideSeek;

    uint32_t num_worlds = 1;
    if (argc >= 2) {
        num_worlds = (uint32_t)atoi(argv[1]);
    }

    ExecMode exec_mode = ExecMode::CPU;
    if (argc >= 3) {
        if (!strcmp("--cpu", argv[2])) {
            exec_mode = ExecMode::CPU;
        } else if (!strcmp("--cuda", argv[2])) {
            exec_mode = ExecMode::CUDA;
        }
    }

    uint32_t num_views = 5;

    const char *replay_log_path = nullptr;
    if (argc >= 4) {
        replay_log_path = argv[3];
    }

    auto replay_log = Optional<HeapArray<int32_t>>::none();
    uint32_t cur_replay_step = 0;
    uint32_t num_replay_steps = 0;
    if (replay_log_path != nullptr) {
        replay_log = readReplayLog(replay_log_path);
        num_replay_steps = replay_log->size() / (num_worlds * num_views * 5);
    }

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
        { math::Vector4{0.1f, 0.1f, 1.0f, 0.0f}, 1, 0.8f, 1.0f,},
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

    math::Quat initial_camera_rotation =
        (math::Quat::angleAxis(-math::pi / 2.f, math::up) *
        math::Quat::angleAxis(-math::pi / 2.f, math::right)).normalize();

    Viewer viewer({
        .gpuID = 0,
        .renderWidth = 1920,
        .renderHeight = 1080,
        .numWorlds = num_worlds,
        .maxViewsPerWorld = num_views,
        .maxInstancesPerWorld = 1000,
        .defaultSimTickRate = 15,
        .cameraMoveSpeed = 10.f,
        .cameraPosition = { 0, 15.f, 30 },
        .cameraRotation = initial_camera_rotation,
        .execMode = exec_mode,
    });

    viewer.loadObjects(render_assets->objects, materials, {
        { (std::filesystem::path(DATA_DIR) /
           "green_grid.png").string().c_str() },
        { (std::filesystem::path(DATA_DIR) /
           "smile.png").string().c_str() },
    });

    viewer.configureLighting({
        { true, math::Vector3{1.0f, 1.0f, -2.f}, math::Vector3{1.0f, 1.0f, 1.0f} }
    });

    Manager mgr({
        .execMode = exec_mode,
        .gpuID = 0,
        .numWorlds = num_worlds,
        .renderWidth = 0,
        .renderHeight = 0,
        .autoReset = replay_log_path != nullptr,
        .enableBatchRender = false,
        .debugCompile = false,
    }, viewer.rendererBridge());

    auto replayStep = [&]() {
        if (cur_replay_step == num_replay_steps - 1) {
            return true;
        }

        printf("Step: %u\n", cur_replay_step);

        for (uint32_t i = 0; i < num_worlds; i++) {
            for (uint32_t j = 0; j < num_views; j++) {
                uint32_t base_idx = 0;
                base_idx = 5 * (cur_replay_step * num_views * num_worlds +
                    i * num_views + j);

                int32_t move_amount = (*replay_log)[base_idx];
                int32_t move_angle = (*replay_log)[base_idx + 1];
                int32_t turn = (*replay_log)[base_idx + 2];
                int32_t g = (*replay_log)[base_idx + 3];
                int32_t l = (*replay_log)[base_idx + 4];

                printf("%d, %d: %d %d %d %d %d\n",
                       i, j, move_amount, move_angle, turn, g, l);
                mgr.setAction(i * num_views + j, move_amount, move_angle, turn, g, l);
            }
        }

        cur_replay_step++;

        return false;
    };

    viewer.loop([&](CountT world_idx, CountT agent_idx,
                       const Viewer::UserInput &input) {
        using Key = Viewer::KeyboardKey;

        int32_t x = 5;
        int32_t y = 5;
        int32_t r = 5;
        bool g = false;
        bool l = false;

        if (input.keyPressed(Key::R)) {
            mgr.triggerReset(world_idx, 1, 3, 2);
        }

        if (input.keyPressed(Key::K1)) {
            mgr.triggerReset(world_idx, 1, 3, 2);
        }

        if (input.keyPressed(Key::K2)) {
            mgr.triggerReset(world_idx, 2, 3, 2);
        }

        if (input.keyPressed(Key::K3)) {
            mgr.triggerReset(world_idx, 3, 3, 2);
        }

        if (input.keyPressed(Key::K4)) {
            mgr.triggerReset(world_idx, 4, 3, 2);
        }

        if (input.keyPressed(Key::K5)) {
            mgr.triggerReset(world_idx, 5, 3, 2);
        }

        if (input.keyPressed(Key::K6)) {
            mgr.triggerReset(world_idx, 6, 3, 2);
        }

        if (input.keyPressed(Key::K7)) {
            mgr.triggerReset(world_idx, 7, 3, 2);
        }

        if (input.keyPressed(Key::K8)) {
            mgr.triggerReset(world_idx, 8, 3, 2);
        }

        if (input.keyPressed(Key::K9)) {
            mgr.triggerReset(world_idx, 9, 3, 2);
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

        mgr.setAction(world_idx * num_views + agent_idx, x, y, r, g, l);
    }, [&]() {
        if (replay_log.has_value()) {
            bool replay_finished = replayStep();

            if (replay_finished) {
                viewer.stopLoop();
            }
        }

        mgr.step();
    }, []() {});
}
