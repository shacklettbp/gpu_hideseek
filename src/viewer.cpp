#include <madrona/viz/viewer.hpp>
#include <madrona/render/render_mgr.hpp>

#include "mgr.hpp"

#include <filesystem>
#include <fstream>

using namespace madrona;
using namespace madrona::viz;

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
    ExecMode exec_mode = ExecMode::CPU;

    auto usageErr = [argv]() {
        fprintf(stderr, "%s [NUM_WORLDS] [--backend cpu|cuda] [--record path] [--replay path] [--load-ckpt path] [--print-obs]\n", argv[0]);
        exit(EXIT_FAILURE);
    };

    bool num_worlds_set = false;

    char *record_log_path = nullptr;
    char *replay_log_path = nullptr;
    char *load_ckpt_path = nullptr;
    bool start_frozen = false;
    bool print_obs = false;

    for (int i = 1; i < argc; i++) {
        char *arg = argv[i];

        if (arg[0] == '-' && arg[1] == '-') {
            arg += 2;

            if (!strcmp("backend", arg)) {
                i += 1;

                if (i == argc) {
                    usageErr();
                }

                char *value = argv[i];
                if (!strcmp("cpu", value)) {
                    exec_mode = ExecMode::CPU;
                } else if (!strcmp("cuda", value)) {
                    exec_mode = ExecMode::CUDA;
                } else {
                    usageErr();
                }
            } else if (!strcmp("record", arg)) {
                if (record_log_path != nullptr) {
                    usageErr();
                }

                i += 1;

                if (i == argc) {
                    usageErr();
                }

                record_log_path = argv[i];
            } else if (!strcmp("replay", arg)) {
                if (replay_log_path != nullptr) {
                    usageErr();
                }

                i += 1;

                if (i == argc) {
                    usageErr();
                }

                replay_log_path = argv[i];
            } else if (!strcmp("load-ckpt", arg)) {
                if (load_ckpt_path != nullptr) {
                    usageErr();
                }

                i += 1;

                if (i == argc) {
                    usageErr();
                }

                load_ckpt_path = argv[i];
            } else if (!strcmp("freeze", arg)) {
                start_frozen = true;
            } else if (!strcmp("print-obs", arg)) {
                print_obs = true;
            } else {
                usageErr();
            }
        } else {
            if (num_worlds_set) {
                usageErr();
            }

            num_worlds_set = true;

            num_worlds = (uint32_t)atoi(arg);
        }
    }

    uint32_t num_views = 5;

    auto replay_log = Optional<HeapArray<int32_t>>::none();
    uint32_t cur_replay_step = 0;
    uint32_t num_replay_steps = 0;
    if (replay_log_path != nullptr) {
        replay_log = readReplayLog(replay_log_path);
        num_replay_steps = replay_log->size() / (num_worlds * num_views * 5);
    }

    
    bool enable_batch_renderer =
#ifdef MADRONA_MACOS
        false;
#else
        true;
#endif

    WindowManager wm {};
    WindowHandle window = wm.makeWindow("Hide & Seek", 2730, 1536);
    render::GPUHandle render_gpu = wm.initGPU(0, { window.get() });

    Manager mgr({
        .execMode = exec_mode,
        .gpuID = 0,
        .numWorlds = num_worlds,
        .autoReset = replay_log_path != nullptr,
        .maxAgentsPerWorld = num_views,
        .enableBatchRenderer = enable_batch_renderer,
        .extRenderAPI = wm.gpuAPIManager().backend(),
        .extRenderDev = render_gpu.device(),
    });

    math::Quat initial_camera_rotation =
        (math::Quat::angleAxis(-math::pi / 2.f, math::up) *
        math::Quat::angleAxis(-math::pi / 2.f, math::right)).normalize();

    viz::Viewer viewer(mgr.getRenderManager(), window.get(), {
        .numWorlds = num_worlds,
        .simTickRate = start_frozen ? 0_u32 : 25_u32,
        .cameraMoveSpeed = 10.f,
        .cameraPosition = { 0, 15.f, 30 },
        .cameraRotation = initial_camera_rotation,
    });

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

    auto global_pos_printer = mgr.globalPositionsTensor().makePrinter();
    auto prep_count_printer = mgr.prepCounterTensor().makePrinter();
    auto vis_agents_printer = mgr.agentDataTensor().makePrinter();
    auto vis_agents_mask_printer = mgr.visibleAgentsMaskTensor().makePrinter();
    auto lidar_printer = mgr.lidarTensor().makePrinter();
    auto reward_printer = mgr.rewardTensor().makePrinter();

    auto printObs = [&]() {
        if (!print_obs) {
            return;
        }

        printf("Global Position\n");
        global_pos_printer.print();
        printf("Prep Counter\n");
        prep_count_printer.print();
        printf("Agents\n");
        vis_agents_printer.print();
        printf("Visible Agents Mask\n");
        vis_agents_mask_printer.print();
        printf("Lidar\n");
        lidar_printer.print();
        printf("Reward\n");
        reward_printer.print();
        

        printf("\n");
    };

    viewer.loop(
    [&](CountT world_idx,
        const Viewer::UserInput &input)
    {
        using Key = Viewer::KeyboardKey;

        if (input.keyHit(Key::R)) {
            mgr.triggerReset(world_idx, 1, 3, 2);
        }

        if (input.keyHit(Key::K1)) {
            mgr.triggerReset(world_idx, 1, 3, 2);
        }

        if (input.keyHit(Key::K2)) {
            mgr.triggerReset(world_idx, 2, 3, 2);
        }

        if (input.keyHit(Key::K3)) {
            mgr.triggerReset(world_idx, 3, 3, 2);
        }

        if (input.keyHit(Key::K4)) {
            mgr.triggerReset(world_idx, 4, 3, 2);
        }

        if (input.keyHit(Key::K5)) {
            mgr.triggerReset(world_idx, 5, 3, 2);
        }

        if (input.keyHit(Key::K6)) {
            mgr.triggerReset(world_idx, 6, 3, 2);
        }

        if (input.keyHit(Key::K7)) {
            mgr.triggerReset(world_idx, 7, 3, 2);
        }

        if (input.keyHit(Key::K8)) {
            mgr.triggerReset(world_idx, 8, 3, 2);
        }

        if (input.keyHit(Key::K9)) {
            mgr.triggerReset(world_idx, 9, 3, 2);
        }

    },
    [&](CountT world_idx, CountT agent_idx,
        const Viewer::UserInput &input)
    {
        using Key = Viewer::KeyboardKey;

        int32_t x = 5;
        int32_t y = 5;
        int32_t r = 5;
        bool g = false;
        bool l = false;

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

        if (input.keyHit(Key::G)) {
            g = true;
        }
        if (input.keyHit(Key::L)) {
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

        printObs();
    }, []() {});
}
