#include "mgr.hpp"
#include "sim.hpp"

#include <madrona/utils.hpp>
#include <madrona/importer.hpp>
#include <madrona/physics_loader.hpp>
#include <madrona/tracing.hpp>
#include <madrona/mw_cpu.hpp>

#include <array>
#include <charconv>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <string>

#ifdef MADRONA_CUDA_SUPPORT
#include <madrona/mw_gpu.hpp>
#include <madrona/cuda_utils.hpp>
#endif

using namespace madrona;
using namespace madrona::math;
using namespace madrona::phys;
using namespace madrona::py;

namespace GPUHideSeek {

struct Manager::Impl {
    Config cfg;
    PhysicsLoader physicsLoader;
    EpisodeManager *episodeMgr;
    WorldReset *resetsPointer;
    Action *actionsPointer;
    float *rewardsBuffer;
    uint8_t *donesBuffer;

    static inline Impl * init(
        const Config &cfg,
        const viz::VizECSBridge *viz_bridge,
        const render::BatchRendererECSBridge *batch_render_bridge);
};

struct Manager::CPUImpl : Manager::Impl {
    using TaskGraphT =
        TaskGraphExecutor<Engine, Sim, GPUHideSeek::Config, WorldInit>;

    TaskGraphT cpuExec;
};

#ifdef MADRONA_CUDA_SUPPORT
struct Manager::CUDAImpl : Manager::Impl {
    MWCudaExecutor mwGPU;
};
#endif

static void loadPhysicsObjects(PhysicsLoader &loader)
{
    SourceCollisionPrimitive sphere_prim {
        .type = CollisionPrimitive::Type::Sphere,
        .sphere = CollisionPrimitive::Sphere {
            .radius = 1.f,
        },
    };

    SourceCollisionPrimitive plane_prim {
        .type = CollisionPrimitive::Type::Plane,
    };

    char import_err_buffer[4096];
    auto imported_hulls = imp::ImportedAssets::importFromDisk({
        (std::filesystem::path(DATA_DIR) / "cube_collision.obj").string().c_str(),
        (std::filesystem::path(DATA_DIR) / "wall_collision.obj").string().c_str(),
        (std::filesystem::path(DATA_DIR) / "agent_collision.obj").string().c_str(),
        (std::filesystem::path(DATA_DIR) / "ramp_collision.obj").string().c_str(),
        (std::filesystem::path(DATA_DIR) / "elongated_collision.obj").string().c_str(),
    }, import_err_buffer, true);

    if (!imported_hulls.has_value()) {
        FATAL("%s", import_err_buffer);
    }

    DynArray<imp::SourceMesh> src_convex_hulls(
        imported_hulls->objects.size());

    DynArray<DynArray<SourceCollisionPrimitive>> prim_arrays(0);
    HeapArray<SourceCollisionObject> src_objs(imported_hulls->objects.size() + 2);

    // Sphere (0)
    src_objs[0] = {
        .prims = Span<const SourceCollisionPrimitive>(&sphere_prim, 1),
        .invMass = 1.f,
        .friction = {
            .muS = 0.5f,
            .muD = 0.5f,
        },
    };

    // Plane (1)
    src_objs[1] = {
        .prims = Span<const SourceCollisionPrimitive>(&plane_prim, 1),
        .invMass = 0.f,
        .friction = {
            .muS = 0.1f,
            .muD = 0.1f,
        },
    };

    auto setupHull = [&](CountT obj_idx, float inv_mass,
                         RigidBodyFrictionData friction) {
        auto meshes = imported_hulls->objects[obj_idx].meshes;
        DynArray<SourceCollisionPrimitive> prims(meshes.size());

        for (const imp::SourceMesh &mesh : meshes) {
            src_convex_hulls.push_back(mesh);
            prims.push_back({
                .type = CollisionPrimitive::Type::Hull,
                .hullInput = {
                    .hullIDX = uint32_t(src_convex_hulls.size() - 1),
                },
            });
        }

        prim_arrays.emplace_back(std::move(prims));

        return SourceCollisionObject {
            .prims = Span<const SourceCollisionPrimitive>(prim_arrays.back()),
            .invMass = inv_mass,
            .friction = friction,
        };
    };

    { // Cube (2)
        src_objs[2] = setupHull(0, 0.5f, {
            .muS = 0.5f,
            .muD = 4.f,
        });
    }

    { // Wall (3)
        src_objs[3] = setupHull(1, 0.f, {
            .muS = 0.5f,
            .muD = 2.f,
        });
    }

    { // Cylinder (4)
        src_objs[4] = setupHull(2, 1.f, {
            .muS = 0.01f,
            .muD = 0.01f,
        });
    }

    { // Ramp (5)
        src_objs[5] = setupHull(3, 0.5f, {
            .muS = 0.5f,
            .muD = 1.f,
        });
    }

    { // Elongated Box (6)
        src_objs[6] = setupHull(4, 0.5f, {
            .muS = 0.5f,
            .muD = 4.f,
        });
    }

    StackAlloc tmp_alloc;
    RigidBodyAssets rigid_body_assets;
    CountT num_rigid_body_data_bytes;
    void *rigid_body_data = RigidBodyAssets::processRigidBodyAssets(
        src_convex_hulls,
        src_objs,
        false,
        tmp_alloc,
        &rigid_body_assets,
        &num_rigid_body_data_bytes);

    if (rigid_body_data == nullptr) {
        FATAL("Invalid collision hull input");
    }

    // HACK:
    rigid_body_assets.metadatas[4].mass.invInertiaTensor.x = 0.f,
    rigid_body_assets.metadatas[4].mass.invInertiaTensor.y = 0.f,

    loader.loadRigidBodies(rigid_body_assets);
    free(rigid_body_data);
}

Manager::Impl * Manager::Impl::init(
    const Config &cfg,
    const viz::VizECSBridge *viz_bridge,
    const render::BatchRendererECSBridge *batch_render_bridge)
{
    HostEventLogging(HostEvent::initStart);

    std::array<char, 1024> import_err;
    auto render_assets = imp::ImportedAssets::importFromDisk({
        (std::filesystem::path(DATA_DIR) / "sphere.obj").string().c_str(),
        (std::filesystem::path(DATA_DIR) / "plane.obj").string().c_str(),
        (std::filesystem::path(DATA_DIR) / "cube_render.obj").string().c_str(),
        (std::filesystem::path(DATA_DIR) / "wall_render.obj").string().c_str(),
        (std::filesystem::path(DATA_DIR) / "cylinder_render.obj").string().c_str(),
        (std::filesystem::path(DATA_DIR) / "ramp_render.obj").string().c_str(),
        (std::filesystem::path(DATA_DIR) / "elongated_render.obj").string().c_str(),
    }, Span<char>(import_err.data(), import_err.size()));

    if (!render_assets.has_value()) {
        FATAL("Failed to load render assets: %s", import_err);
    }

    GPUHideSeek::Config app_cfg {
        batch_render_bridge != nullptr,
        viz_bridge != nullptr,
        cfg.autoReset,
    };

    switch (cfg.execMode) {
    case ExecMode::CUDA: {
#ifdef MADRONA_CUDA_SUPPORT
        CUcontext cu_ctx = MWCudaExecutor::initCUDA(cfg.gpuID);

        EpisodeManager *episode_mgr = 
            (EpisodeManager *)cu::allocGPU(sizeof(EpisodeManager));
        REQ_CUDA(cudaMemset(episode_mgr, 0, sizeof(EpisodeManager)));

        PhysicsLoader phys_loader(cfg.execMode, 10);
        loadPhysicsObjects(phys_loader);

        ObjectManager *phys_obj_mgr = &phys_loader.getObjectManager();

        auto done_buffer = (uint8_t *)cu::allocGPU(sizeof(uint8_t) *
            consts::maxAgents * cfg.numWorlds);

        auto reward_buffer = (float *)cu::allocGPU(sizeof(float) *
            consts::maxAgents * cfg.numWorlds);

        HeapArray<WorldInit> world_inits(cfg.numWorlds);

        for (int64_t i = 0; i < (int64_t)cfg.numWorlds; i++) {
            world_inits[i] = WorldInit {
                episode_mgr,
                reward_buffer,
                done_buffer,
                phys_obj_mgr,
                0,
                1,
                viz_bridge,
                batch_render_bridge,
            };
        }

        MWCudaExecutor mwgpu_exec({
            .worldInitPtr = world_inits.data(),
            .numWorldInitBytes = sizeof(WorldInit),
            .userConfigPtr = &app_cfg,
            .numUserConfigBytes = sizeof(GPUHideSeek::Config),
            .numWorldDataBytes = sizeof(Sim),
            .worldDataAlignment = alignof(Sim),
            .numWorlds = cfg.numWorlds,
            .numExportedBuffers = 16,
        }, {
            { GPU_HIDESEEK_SRC_LIST },
            { GPU_HIDESEEK_COMPILE_FLAGS },
            cfg.debugCompile ? CompileConfig::OptMode::Debug :
                CompileConfig::OptMode::LTO,
        }, cu_ctx);

        WorldReset *world_reset_buffer = 
            (WorldReset *)mwgpu_exec.getExported(0);

        Action *agent_actions_buffer = 
            (Action *)mwgpu_exec.getExported(3);

        HostEventLogging(HostEvent::initEnd);
        return new CUDAImpl {
            { 
                cfg,
                std::move(phys_loader),
                episode_mgr,
                world_reset_buffer,
                agent_actions_buffer,
                reward_buffer,
                done_buffer,
            },
            std::move(mwgpu_exec),
        };
#else
        FATAL("Madrona was not compiled with CUDA support");
#endif
    } break;
    case ExecMode::CPU: {
        EpisodeManager *episode_mgr = new EpisodeManager { 0 };

        PhysicsLoader phys_loader(cfg.execMode, 10);
        loadPhysicsObjects(phys_loader);

        ObjectManager *phys_obj_mgr = &phys_loader.getObjectManager();

        auto reward_buffer = (float *)malloc(
            sizeof(float) * consts::maxAgents * cfg.numWorlds);

        auto done_buffer = (uint8_t *)malloc(
            sizeof(uint8_t) * consts::maxAgents * cfg.numWorlds);

        HeapArray<WorldInit> world_inits(cfg.numWorlds);

        for (int64_t i = 0; i < (int64_t)cfg.numWorlds; i++) {
            world_inits[i] = WorldInit {
                episode_mgr,
                reward_buffer + i * consts::maxAgents,
                done_buffer + i * consts::maxAgents,
                phys_obj_mgr,
                0, 0,
                viz_bridge,
                batch_render_bridge,
            };
        }

        CPUImpl::TaskGraphT cpu_exec {
            ThreadPoolExecutor::Config {
                .numWorlds = cfg.numWorlds,
                .numExportedBuffers = 16,
            },
            app_cfg,
            world_inits.data(),
        };

        WorldReset *world_reset_buffer =
            (WorldReset *)cpu_exec.getExported(0);

        Action *agent_actions_buffer = 
            (Action *)cpu_exec.getExported(3);

        auto cpu_impl = new CPUImpl {
            { 
                cfg,
                std::move(phys_loader),
                episode_mgr,
                world_reset_buffer,
                agent_actions_buffer,
                reward_buffer,
                done_buffer,
            },
            std::move(cpu_exec),
        };

        HostEventLogging(HostEvent::initEnd);

        return cpu_impl;
    } break;
    default: MADRONA_UNREACHABLE();
    }
}

Manager::Manager(
        const Config &cfg,
        const madrona::viz::VizECSBridge *viz_bridge,
        const madrona::render::BatchRendererECSBridge *batch_render_bridge)
    : impl_(Impl::init(cfg, viz_bridge, batch_render_bridge))
{
    for (int32_t i = 0; i < (int32_t)cfg.numWorlds; i++) {
        triggerReset(i, 1, 3, 2);
    }

    step();
}

Manager::~Manager() {
    switch (impl_->cfg.execMode) {
    case ExecMode::CUDA: {
#ifdef MADRONA_CUDA_SUPPORT
        delete static_cast<CUDAImpl *>(impl_);
#endif
    } break;
    case ExecMode::CPU : {
        delete static_cast<CPUImpl *>(impl_);
    } break;
    }
#ifdef MADRONA_TRACING
    FinalizeLogging("/tmp/");
#endif
}

void Manager::step()
{
    switch (impl_->cfg.execMode) {
    case ExecMode::CUDA: {
#ifdef MADRONA_CUDA_SUPPORT
        static_cast<CUDAImpl *>(impl_)->mwGPU.run();
#endif
    } break;
    case ExecMode::CPU: {
        auto cpu_impl = static_cast<CPUImpl *>(impl_);
        cpu_impl->cpuExec.run();

        // FIXME: provide some way to do this more cleanly in madrona
        CountT cur_agent_offset = 0;
        float *base_rewards = cpu_impl->rewardsBuffer;
        uint8_t *base_dones = cpu_impl->donesBuffer;

        for (CountT i = 0; i < (CountT)impl_->cfg.numWorlds; i++) {
            const Sim &sim_data = cpu_impl->cpuExec.getWorldData(i);
            CountT num_agents = sim_data.numActiveAgents;
            float *world_rewards = sim_data.rewardBuffer;
            uint8_t *world_dones = sim_data.doneBuffer;

            memmove(&base_rewards[cur_agent_offset],
                    world_rewards,
                    sizeof(float) * num_agents);

            memmove(&base_dones[cur_agent_offset],
                    world_dones,
                    sizeof(uint8_t) * num_agents);

            cur_agent_offset += num_agents;
        }
    } break;
    }
}


Tensor Manager::resetTensor() const
{
    return exportStateTensor(0, Tensor::ElementType::Int32,
                             {impl_->cfg.numWorlds, 3});
}

Tensor Manager::doneTensor() const
{
    Optional<int> gpu_id = Optional<int>::none();
    if (impl_->cfg.execMode == ExecMode::CUDA) {
        gpu_id = impl_->cfg.gpuID;
    }

    return Tensor(impl_->donesBuffer, Tensor::ElementType::UInt8,
                 {impl_->cfg.numWorlds * consts::maxAgents, 1}, gpu_id);
}

madrona::py::Tensor Manager::prepCounterTensor() const
{
    return exportStateTensor(2, Tensor::ElementType::Int32,
                             {impl_->cfg.numWorlds * consts::maxAgents, 1});
}

Tensor Manager::actionTensor() const
{
    return exportStateTensor(3, Tensor::ElementType::Int32,
                             {impl_->cfg.numWorlds * consts::maxAgents, 5});
}

Tensor Manager::rewardTensor() const
{
    Optional<int> gpu_id = Optional<int>::none();
    if (impl_->cfg.execMode == ExecMode::CUDA) {
        gpu_id = impl_->cfg.gpuID;
    }

    return Tensor(impl_->rewardsBuffer, Tensor::ElementType::Float32,
                 {impl_->cfg.numWorlds * consts::maxAgents, 1}, gpu_id);
}

Tensor Manager::agentTypeTensor() const
{
    return exportStateTensor(5, Tensor::ElementType::Int32,
                             {impl_->cfg.numWorlds * consts::maxAgents, 1});
}

Tensor Manager::agentMaskTensor() const
{
    return exportStateTensor(6, Tensor::ElementType::Float32,
                             {impl_->cfg.numWorlds * consts::maxAgents, 1});
}


madrona::py::Tensor Manager::agentDataTensor() const
{
    return exportStateTensor(7, Tensor::ElementType::Float32,
                             {
                                 impl_->cfg.numWorlds * consts::maxAgents,
                                 consts::maxAgents - 1,
                                 4,
                             });
}

madrona::py::Tensor Manager::boxDataTensor() const
{
    return exportStateTensor(8, Tensor::ElementType::Float32,
                             {
                                 impl_->cfg.numWorlds * consts::maxAgents,
                                 consts::maxBoxes,
                                 7,
                             });
}

madrona::py::Tensor Manager::rampDataTensor() const
{
    return exportStateTensor(9, Tensor::ElementType::Float32,
                             {
                                 impl_->cfg.numWorlds * consts::maxAgents,
                                 consts::maxRamps,
                                 5,
                             });
}

madrona::py::Tensor Manager::visibleAgentsMaskTensor() const
{
    return exportStateTensor(10, Tensor::ElementType::Float32,
                             {
                                 impl_->cfg.numWorlds * consts::maxAgents,
                                 consts::maxAgents - 1,
                                 1,
                             });
}

madrona::py::Tensor Manager::visibleBoxesMaskTensor() const
{
    return exportStateTensor(11, Tensor::ElementType::Float32,
                             {
                                 impl_->cfg.numWorlds * consts::maxAgents,
                                 consts::maxBoxes,
                                 1,
                             });
}

madrona::py::Tensor Manager::visibleRampsMaskTensor() const
{
    return exportStateTensor(12, Tensor::ElementType::Float32,
                             {
                                 impl_->cfg.numWorlds * consts::maxAgents,
                                 consts::maxRamps,
                                 1,
                             });
}

madrona::py::Tensor Manager::globalPositionsTensor() const
{
    return exportStateTensor(13, Tensor::ElementType::Float32,
                             {
                                 impl_->cfg.numWorlds,
                                 consts::maxBoxes + consts::maxRamps +
                                     consts::maxAgents,
                                 2,
                             });
}

Tensor Manager::depthTensor() const
{
    void *dev_ptr = nullptr;
    Optional<int> gpu_id = Optional<int>::none();

#if 0
    if (impl_->cfg.execMode == ExecMode::CUDA) {
#ifdef MADRONA_CUDA_SUPPORT
        dev_ptr = static_cast<CUDAImpl *>(impl_)->mwGPU.
            depthObservations();
        gpu_id = impl_->cfg.gpuID;
#endif
    } else {
        dev_ptr = static_cast<CPUImpl *>(impl_)->cpuExec.
            depthObservations();

#ifdef MADRONA_LINUX
        gpu_id = impl_->cfg.gpuID;
#endif
    }
#endif

    return Tensor(dev_ptr, Tensor::ElementType::Float32,
                     {impl_->cfg.numWorlds * consts::maxAgents,
                      impl_->cfg.renderHeight,
                      impl_->cfg.renderWidth, 1}, gpu_id);
}

Tensor Manager::rgbTensor() const
{
    void *dev_ptr = nullptr;
    Optional<int> gpu_id = Optional<int>::none();

#if 0
    if (impl_->cfg.execMode == ExecMode::CUDA) {
#ifdef MADRONA_CUDA_SUPPORT
        dev_ptr = static_cast<CUDAImpl *>(impl_)->mwGPU.
            rgbObservations();
        gpu_id = impl_->cfg.gpuID;
#endif
    } else {
        dev_ptr = static_cast<CPUImpl *>(impl_)->cpuExec.
            rgbObservations();

#ifdef MADRONA_LINUX
        gpu_id = impl_->cfg.gpuID;
#endif
    }
#endif

    return Tensor(dev_ptr, Tensor::ElementType::UInt8,
                  {impl_->cfg.numWorlds * consts::maxAgents,
                   impl_->cfg.renderHeight,
                   impl_->cfg.renderWidth, 4}, gpu_id);
}

madrona::py::Tensor Manager::lidarTensor() const
{
    return exportStateTensor(14, Tensor::ElementType::Float32,
                             {
                                 impl_->cfg.numWorlds * consts::maxAgents,
                                 30,
                             });
}

madrona::py::Tensor Manager::seedTensor() const
{
    return exportStateTensor(15, Tensor::ElementType::Int32,
                             {
                                 impl_->cfg.numWorlds * consts::maxAgents,
                                 1,
                             });
}

void Manager::triggerReset(CountT world_idx, CountT level_idx,
                           CountT num_hiders, CountT num_seekers)
{
    WorldReset reset {
        (int32_t)level_idx,
        (int32_t)num_hiders,
        (int32_t)num_seekers,
    };

    auto *reset_ptr = impl_->resetsPointer + world_idx;

    if (impl_->cfg.execMode == ExecMode::CUDA) {
#ifdef MADRONA_CUDA_SUPPORT
        cudaMemcpy(reset_ptr, &reset, sizeof(WorldReset),
                   cudaMemcpyHostToDevice);
#endif
    }  else {
        *reset_ptr = reset;
    }
}

void Manager::setAction(CountT agent_idx,
                        int32_t x, int32_t y, int32_t r,
                        bool g, bool l)
{
    Action action { 
        .x = x,
        .y = y,
        .r = r,
        .g = (int32_t)g,
        .l = (int32_t)l,
    };

    auto *action_ptr = impl_->actionsPointer + agent_idx;

    if (impl_->cfg.execMode == ExecMode::CUDA) {
#ifdef MADRONA_CUDA_SUPPORT
        cudaMemcpy(action_ptr, &action, sizeof(Action),
                   cudaMemcpyHostToDevice);
#endif
    } else {
        *action_ptr = action;
    }
}

Tensor Manager::exportStateTensor(int64_t slot,
                                  Tensor::ElementType type,
                                  Span<const int64_t> dimensions) const
{
    void *dev_ptr = nullptr;
    Optional<int> gpu_id = Optional<int>::none();
    if (impl_->cfg.execMode == ExecMode::CUDA) {
#ifdef MADRONA_CUDA_SUPPORT
        dev_ptr =
            static_cast<CUDAImpl *>(impl_)->mwGPU.getExported(slot);
        gpu_id = impl_->cfg.gpuID;
#endif
    } else {
        dev_ptr = static_cast<CPUImpl *>(impl_)->cpuExec.getExported(slot);
    }

    return Tensor(dev_ptr, type, dimensions, gpu_id);
}


}
