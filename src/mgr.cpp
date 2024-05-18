#include "mgr.hpp"
#include "sim.hpp"

#include <madrona/utils.hpp>
#include <madrona/importer.hpp>
#include <madrona/physics_loader.hpp>
#include <madrona/tracing.hpp>
#include <madrona/mw_cpu.hpp>
#include <madrona/render/api.hpp>

#include <array>
#include <charconv>
#include <chrono>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <string>

#include <glm/gtx/string_cast.hpp>

#ifdef MADRONA_CUDA_SUPPORT
#include <madrona/mw_gpu.hpp>
#include <madrona/cuda_utils.hpp>
#endif

#include <bps3D.hpp>
#include <bps3D_madrona.hpp>

#include <stb_image_write.h>

using namespace madrona;
using namespace madrona::math;
using namespace madrona::phys;
using namespace madrona::py;

using bps3D::BPS3DState;

namespace GPUHideSeek {

struct RenderGPUState {
    render::APILibHandle apiLib;
    render::APIManager apiMgr;
    render::GPUHandle gpu;
};

static inline Optional<RenderGPUState> initRenderGPUState(
    const Manager::Config &mgr_cfg)
{
    if (mgr_cfg.useBPS3D) {
        return Optional<RenderGPUState>::none();
    }

    if (!mgr_cfg.headlessMode) {
        if (mgr_cfg.extRenderDev || !mgr_cfg.enableBatchRenderer) {
            return Optional<RenderGPUState>::none();
        }
    }

    auto render_api_lib = render::APIManager::loadDefaultLib();
    render::APIManager render_api_mgr(render_api_lib.lib());
    render::GPUHandle gpu = render_api_mgr.initGPU(mgr_cfg.gpuID);

    return RenderGPUState {
        .apiLib = std::move(render_api_lib),
        .apiMgr = std::move(render_api_mgr),
        .gpu = std::move(gpu),
    };
}

static inline Optional<render::RenderManager> initRenderManager(
    const Manager::Config &mgr_cfg,
    const Optional<RenderGPUState> &render_gpu_state)
{
    if (mgr_cfg.headlessMode && !mgr_cfg.enableBatchRenderer) {
        return Optional<render::RenderManager>::none();
    }

    if (!mgr_cfg.headlessMode) {
        if (!mgr_cfg.extRenderDev && !mgr_cfg.enableBatchRenderer) {
            return Optional<render::RenderManager>::none();
        }
    }

    render::APIBackend *render_api;
    render::GPUDevice *render_dev;

    if (render_gpu_state.has_value()) {
        render_api = render_gpu_state->apiMgr.backend();
        render_dev = render_gpu_state->gpu.device();
    } else {
        render_api = mgr_cfg.extRenderAPI;
        render_dev = mgr_cfg.extRenderDev;
    }

    return render::RenderManager(render_api, render_dev, {
        .enableBatchRenderer = mgr_cfg.enableBatchRenderer,
        .agentViewWidth = mgr_cfg.batchRenderViewWidth,
        .agentViewHeight = mgr_cfg.batchRenderViewHeight,
        .numWorlds = mgr_cfg.numWorlds,
        .maxViewsPerWorld = mgr_cfg.maxHiders + mgr_cfg.maxSeekers,
        .maxInstancesPerWorld = 1000,
        .execMode = mgr_cfg.execMode,
        .voxelCfg = {},
    });
}

static inline Optional<BPS3DState> initBPS3D(
    const Manager::Config &mgr_cfg)
{
    if (!mgr_cfg.useBPS3D) {
        return Optional<BPS3DState>::none();
    }

    return bps3D::initBPS3D(mgr_cfg.numWorlds,
        mgr_cfg.batchRenderViewWidth,
        mgr_cfg.batchRenderViewHeight,
        (std::filesystem::path(DATA_DIR) / "bps3d.bps").string().c_str());
}

struct Manager::Impl {
    Config cfg;
    int32_t maxAgentsPerWorld;
    PhysicsLoader physicsLoader;
    Optional<RenderGPUState> renderGPUState;
    Optional<render::RenderManager> renderMgr;
    Optional<BPS3DState> bps3DState;
    WorldReset *resetsPointer;
    Action *actionsPointer;
    uint32_t raycastOutputResolution;
    bool headlessMode;
    BPSBridge bpsBridge;
    double avgBPSTime;
    int64_t numBPSSteps;

    double avgDebugTime { 0.0 };
    int64_t numDebugSteps { -1 };

    static inline Impl * make(const Config &cfg);


    template <EnumType EnumT>
    Tensor exportStateTensor(EnumT slot,
                             TensorElementType type,
                             Span<const int64_t> dimensions);


    inline void bpsRender();
};

struct Manager::CPUImpl : Manager::Impl {
    using TaskGraphT =
        TaskGraphExecutor<Engine, Sim, GPUHideSeek::Config, WorldInit>;

    TaskGraphT cpuExec;

    inline void init();
    inline void step();
};

void Manager::CPUImpl::init()
{
    cpuExec.runTaskGraph(TaskGraphID::Init);
}

void Manager::CPUImpl::step()
{
    cpuExec.runTaskGraph(TaskGraphID::Step);
}

#ifdef MADRONA_CUDA_SUPPORT
struct Manager::CUDAImpl : Manager::Impl {
    MWCudaExecutor mwGPU;
    MWCudaLaunchGraph stepGraph;
    MWCudaLaunchGraph renderGraph;

    inline ~CUDAImpl()
    {
        printf("Time : %f\n", avgBPSTime);
        printf("Debug Time : %f\n", avgDebugTime);

        printf("FPS: %f\n", double(cfg.numWorlds) / (avgBPSTime / 1000.0));
        printf("Debug FPS: %f\n", double(cfg.numWorlds) / (avgDebugTime / 1000.0));
    }

    inline void init();
    inline void step();
};

void Manager::CUDAImpl::init()
{
    MWCudaLaunchGraph init_graph = mwGPU.buildLaunchGraph(TaskGraphID::Init,
                                                          false);

    mwGPU.run(init_graph);
}

void Manager::CUDAImpl::step()
{
    mwGPU.run(stepGraph);
    if (!bps3DState.has_value()) {
        mwGPU.run(renderGraph);
    }
}
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
            .muS = 2.f,
            .muD = 2.f,
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
            .muD = 2.f,
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
            .muS = 0.5f,
            .muD = 16.f,
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

static imp::ImportedAssets loadRenderObjects(
        Optional<render::RenderManager> &render_mgr,
        std::vector<imp::SourceMaterial> materials,
        std::vector<imp::SourceTexture> texture_paths)
{
    std::array<std::string, (size_t)SimObject::NumObjects> render_asset_paths;
    render_asset_paths[(size_t)SimObject::Sphere] =
        (std::filesystem::path(DATA_DIR) / "sphere.obj").string();
    render_asset_paths[(size_t)SimObject::Plane] =
        (std::filesystem::path(DATA_DIR) / "plane.obj").string();
    render_asset_paths[(size_t)SimObject::Cube] =
        (std::filesystem::path(DATA_DIR) / "cube_render.obj").string();
    render_asset_paths[(size_t)SimObject::Wall] =
        (std::filesystem::path(DATA_DIR) / "wall_render.obj").string();
    render_asset_paths[(size_t)SimObject::Agent] =
        (std::filesystem::path(DATA_DIR) / "agent_render.obj").string();
    render_asset_paths[(size_t)SimObject::Ramp] =
        (std::filesystem::path(DATA_DIR) / "ramp_render.obj").string();
    render_asset_paths[(size_t)SimObject::Box] =
        (std::filesystem::path(DATA_DIR) / "elongated_render.obj").string();

    std::array<const char *, (size_t)SimObject::NumObjects> render_asset_cstrs;
    for (size_t i = 0; i < render_asset_paths.size(); i++) {
        render_asset_cstrs[i] = render_asset_paths[i].c_str();
    }

    std::array<char, 1024> import_err;
    auto render_assets = imp::ImportedAssets::importFromDisk(
        render_asset_cstrs, Span<char>(import_err.data(), import_err.size()),
        true, true);

    if (!render_assets.has_value()) {
        FATAL("Failed to load render assets: %s", import_err);
    }

    // Override materials
    render_assets->objects[0].meshes[0].materialIDX = 0;
    render_assets->geoData.meshBVHArrays[0][0].materialIDX = 0;

    render_assets->objects[1].meshes[0].materialIDX = 3;
    render_assets->geoData.meshBVHArrays[1][0].materialIDX = 3;

    render_assets->objects[2].meshes[0].materialIDX = 1;
    render_assets->geoData.meshBVHArrays[2][0].materialIDX = 1;

    render_assets->objects[3].meshes[0].materialIDX = 0;
    render_assets->geoData.meshBVHArrays[3][0].materialIDX = 0;

    render_assets->objects[4].meshes[0].materialIDX = 2;
    render_assets->geoData.meshBVHArrays[4][0].materialIDX = 2;

    render_assets->objects[4].meshes[1].materialIDX = 6;
    render_assets->objects[4].meshes[2].materialIDX = 6;

    render_assets->objects[5].meshes[0].materialIDX = 4;
    render_assets->geoData.meshBVHArrays[5][0].materialIDX = 4;

    render_assets->objects[6].meshes[0].materialIDX = 5;
    render_assets->geoData.meshBVHArrays[6][0].materialIDX = 5;

    if (render_mgr.has_value()) {
        render_mgr->loadObjects(render_assets->objects,
                Span(materials.data(), materials.size()), 
                Span(texture_paths.data(), (CountT)texture_paths.size()),
                true);

        render_mgr->configureLighting({
            { true, math::Vector3{1.0f, 1.0f, -2.0f}, math::Vector3{1.0f, 1.0f, 1.0f} }
        });
    }

    return std::move(*render_assets);
}

Manager::Impl * Manager::Impl::make(const Config &cfg)
{
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

    GPUHideSeek::Config app_cfg;
    app_cfg.simFlags = cfg.simFlags;
    app_cfg.initRandKey = rand::initKey(cfg.randSeed);
    app_cfg.minHiders = cfg.minHiders;
    app_cfg.maxHiders = cfg.maxHiders;
    app_cfg.minSeekers = cfg.minSeekers;
    app_cfg.maxSeekers = cfg.maxSeekers;

    int32_t max_agents_per_world = cfg.maxHiders + cfg.maxSeekers;

    switch (cfg.execMode) {
    case ExecMode::CUDA: {
#ifdef MADRONA_CUDA_SUPPORT
        CUcontext cu_ctx = MWCudaExecutor::initCUDA(cfg.gpuID);

        PhysicsLoader phys_loader(cfg.execMode, 10);
        loadPhysicsObjects(phys_loader);

        ObjectManager *phys_obj_mgr = &phys_loader.getObjectManager();
        app_cfg.rigidBodyObjMgr = phys_obj_mgr;

        Optional<RenderGPUState> render_gpu_state =
            initRenderGPUState(cfg);

        Optional<render::RenderManager> render_mgr =
            initRenderManager(cfg, render_gpu_state);

        imp::ImportedAssets::GPUGeometryData gpu_imported_assets;

        std::vector<imp::SourceMaterial> materials = {
            { math::Vector4{0.4f, 0.4f, 0.4f, 0.0f}, -1, 0.8f, 0.2f,},
            { math::Vector4{1.0f, 0.1f, 0.1f, 0.0f}, -1, 0.8f, 0.2f,},
            { math::Vector4{0.1f, 0.1f, 1.0f, 0.0f}, 1, 0.8f, 1.0f,},
            { math::Vector4{0.5f, 0.3f, 0.3f, 0.0f},  0, 0.8f, 0.2f,},
            { render::rgb8ToFloat(191, 108, 10), -1, 0.8f, 0.2f },
            { render::rgb8ToFloat(12, 144, 150), -1, 0.8f, 0.2f },
            { render::rgb8ToFloat(230, 230, 230),   -1, 0.8f, 1.0f },
        };

        auto green_grid_str = (std::filesystem::path(DATA_DIR) /
               "green_grid.png").string();
        auto smile_str = (std::filesystem::path(DATA_DIR) /
               "smile.png").string();

        std::vector<imp::SourceTexture> textures =  {
            { green_grid_str.c_str() },
            { smile_str.c_str() },
            { smile_str.c_str() }
        };

        auto imported_assets = loadRenderObjects(
                render_mgr, materials, textures);

        auto gpu_imported_assets_opt =
            imp::ImportedAssets::makeGPUData(imported_assets);

        assert(gpu_imported_assets_opt.has_value());

        gpu_imported_assets = std::move(*gpu_imported_assets_opt);

        Optional<BPS3DState> bps3D_state = initBPS3D(cfg);

        if (render_mgr.has_value()) {
            app_cfg.renderBridge = render_mgr->bridge();
        } else {
            app_cfg.renderBridge = nullptr;
        }


        BPSBridge bps_bridge {};

        if (bps3D_state.has_value()) {
            CountT max_render_entities_per_world =
                consts::maxBoxes + consts::maxRamps +
                consts::maxAgents + 30;

            app_cfg.bpsBridge = bps3D::initBridge(bps_bridge, cfg.numWorlds,
                                                  max_render_entities_per_world);
        } else {
            app_cfg.bpsBridge = nullptr;
        }

        HeapArray<WorldInit> world_inits(cfg.numWorlds);

        MWCudaExecutor mwgpu_exec({
            .worldInitPtr = world_inits.data(),
            .numWorldInitBytes = sizeof(WorldInit),
            .userConfigPtr = &app_cfg,
            .numUserConfigBytes = sizeof(GPUHideSeek::Config),
            .numWorldDataBytes = sizeof(Sim),
            .worldDataAlignment = alignof(Sim),
            .numWorlds = cfg.numWorlds,
            .numTaskGraphs = (uint32_t)TaskGraphID::NumTaskGraphs,
            .numExportedBuffers = (uint32_t)ExportID::NumExports,
            .geometryData = &gpu_imported_assets,
            .materials = { materials.data(), (CountT)materials.size() },
            .textures = { textures.data(), (CountT)textures.size() },
            .raycastOutputResolution = cfg.raycastOutputResolution,
            .nearSphere = 2.1f
        }, {
            { GPU_HIDESEEK_SRC_LIST },
            { GPU_HIDESEEK_COMPILE_FLAGS },
            CompileConfig::OptMode::LTO,
        }, cu_ctx);

        MWCudaLaunchGraph step_graph = mwgpu_exec.buildLaunchGraph(
            TaskGraphID::Step, false);

        MWCudaLaunchGraph render_graph = mwgpu_exec.buildLaunchGraph(
            TaskGraphID::Render, !cfg.enableBatchRenderer && !cfg.useBPS3D,
            "render_sort");

        WorldReset *world_reset_buffer = 
            (WorldReset *)mwgpu_exec.getExported((uint32_t)ExportID::Reset);

        Action *agent_actions_buffer = 
            (Action *)mwgpu_exec.getExported((uint32_t)ExportID::Action);


        HostEventLogging(HostEvent::initEnd);
        return new CUDAImpl {
            { 
                .cfg = cfg,
                .maxAgentsPerWorld = max_agents_per_world,
                .physicsLoader = std::move(phys_loader),
                .renderGPUState = std::move(render_gpu_state),
                .renderMgr = std::move(render_mgr),
                .bps3DState = std::move(bps3D_state),
                .resetsPointer = world_reset_buffer,
                .actionsPointer = agent_actions_buffer,
                .raycastOutputResolution = cfg.raycastOutputResolution,
                .headlessMode = cfg.headlessMode,
                .bpsBridge = std::move(bps_bridge),
            },
            std::move(mwgpu_exec),
            std::move(step_graph),
            std::move(render_graph)
        };
#else
        FATAL("Madrona was not compiled with CUDA support");
#endif
    } break;
    case ExecMode::CPU: {
        PhysicsLoader phys_loader(cfg.execMode, 10);
        loadPhysicsObjects(phys_loader);

        ObjectManager *phys_obj_mgr = &phys_loader.getObjectManager();
        app_cfg.rigidBodyObjMgr = phys_obj_mgr;

        Optional<RenderGPUState> render_gpu_state =
            initRenderGPUState(cfg);

        Optional<render::RenderManager> render_mgr =
            initRenderManager(cfg, render_gpu_state);

        if (render_mgr.has_value()) {
            loadRenderObjects(render_mgr, {}, {});
            app_cfg.renderBridge = render_mgr->bridge();
         } else {
            app_cfg.renderBridge = nullptr;
         }

        HeapArray<WorldInit> world_inits(cfg.numWorlds);

        CPUImpl::TaskGraphT cpu_exec {
            ThreadPoolExecutor::Config {
                .numWorlds = cfg.numWorlds,
                .numExportedBuffers = (uint32_t)ExportID::NumExports,
            },
            app_cfg,
            world_inits.data(),
            (uint32_t)TaskGraphID::NumTaskGraphs,
        };

        WorldReset *world_reset_buffer =
            (WorldReset *)cpu_exec.getExported((uint32_t)ExportID::Reset);

        Action *agent_actions_buffer = 
            (Action *)cpu_exec.getExported((uint32_t)ExportID::Action);

        auto cpu_impl = new CPUImpl {
            { 
                cfg,
                max_agents_per_world,
                std::move(phys_loader),
                std::move(render_gpu_state),
                std::move(render_mgr),
                Optional<BPS3DState>::none(),
                world_reset_buffer,
                agent_actions_buffer,
                cfg.raycastOutputResolution,
                !cfg.enableBatchRenderer,
                BPSBridge {},
                0.0, -1,
            },
            std::move(cpu_exec),
        };

        HostEventLogging(HostEvent::initEnd);

        return cpu_impl;
    } break;
    default: MADRONA_UNREACHABLE();
    }
}

template <EnumType EnumT>
Tensor Manager::Impl::exportStateTensor(EnumT slot,
                                        TensorElementType type,
                                        Span<const int64_t> dimensions)
{
    void *dev_ptr = nullptr;
    Optional<int> gpu_id = Optional<int>::none();
    if (cfg.execMode == ExecMode::CUDA) {
#ifdef MADRONA_CUDA_SUPPORT
        dev_ptr =
            static_cast<CUDAImpl *>(this)->mwGPU.getExported((uint32_t)slot);
        gpu_id = cfg.gpuID;
#endif
    } else {
        dev_ptr = static_cast<CPUImpl *>(this)->cpuExec.getExported((uint32_t)slot);
    }

    return Tensor(dev_ptr, type, dimensions, gpu_id);
}

Manager::Manager(const Config &cfg)
    : impl_(Impl::make(cfg))
{}

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
}

void Manager::Impl::bpsRender()
{
    auto start = std::chrono::steady_clock::now();

    static_cast<CUDAImpl *>(this)->mwGPU.run(
        static_cast<CUDAImpl *>(this)->renderGraph);

    bps3D::bpsRender(bps3DState, bpsBridge, cfg.numWorlds);

    auto end = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();

    if (numBPSSteps == -1) {
        numBPSSteps = 0;
    } else {
        avgBPSTime = avgBPSTime * (double(numBPSSteps) /
            double(numBPSSteps + 1)) + 
            elapsed * (1.0 / double(numBPSSteps + 1));
        numBPSSteps += 1;
    }
}

void Manager::init()
{
    switch (impl_->cfg.execMode) {
    case ExecMode::CUDA: {
#ifdef MADRONA_CUDA_SUPPORT
        static_cast<CUDAImpl *>(impl_)->init();
#endif
    } break;
    case ExecMode::CPU: {
        static_cast<CPUImpl *>(impl_)->init();
    } break;
    }

    if (impl_->bps3DState.has_value()) {
        impl_->bpsRender();
    } else {
        if (impl_->headlessMode) {
            if (impl_->cfg.enableBatchRenderer) {
                impl_->renderMgr->readECS();
            }
        } else {
            if (impl_->renderMgr.has_value()) {
                impl_->renderMgr->readECS();
            }
        }

        if (impl_->cfg.enableBatchRenderer) {
            impl_->renderMgr->batchRender();
        }
    }
}

void Manager::step()
{
    auto start = std::chrono::steady_clock::now();

    switch (impl_->cfg.execMode) {
    case ExecMode::CUDA: {
#ifdef MADRONA_CUDA_SUPPORT
        static_cast<CUDAImpl *>(impl_)->step();
#endif
    } break;
    case ExecMode::CPU: {
        static_cast<CPUImpl *>(impl_)->step();
    } break;
    }

    auto end = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();

    if (impl_->numDebugSteps == -1) {
        impl_->numDebugSteps = 0;
    } else {
        impl_->avgDebugTime = impl_->avgDebugTime * (double(impl_->numDebugSteps) /
            double(impl_->numDebugSteps + 1)) + 
            elapsed * (1.0 / double(impl_->numDebugSteps + 1));
        impl_->numDebugSteps += 1;
    }


    if (impl_->bps3DState.has_value()) {
        impl_->bpsRender();
    } else {
        if (impl_->headlessMode) {
            if (impl_->cfg.enableBatchRenderer) {
                impl_->renderMgr->readECS();
            }
        } else {
            if (impl_->renderMgr.has_value()) {
                impl_->renderMgr->readECS();
            }
        }

        if (impl_->cfg.enableBatchRenderer) {
            impl_->renderMgr->batchRender();
        }
    }

}


Tensor Manager::resetTensor() const
{
    return impl_->exportStateTensor(
        ExportID::Reset, TensorElementType::Int32,
        {impl_->cfg.numWorlds, 1});
}

Tensor Manager::doneTensor() const
{
    return impl_->exportStateTensor(
        ExportID::Done, TensorElementType::Int32,
        {impl_->cfg.numWorlds, 1});
}

madrona::py::Tensor Manager::prepCounterTensor() const
{
    return impl_->exportStateTensor(
        ExportID::PrepCounter, TensorElementType::Int32,
        {impl_->cfg.numWorlds * impl_->maxAgentsPerWorld, 1});
}

Tensor Manager::actionTensor() const
{
    return impl_->exportStateTensor(
        ExportID::Action, TensorElementType::Int32,
        {impl_->cfg.numWorlds * impl_->maxAgentsPerWorld, 5});
}

Tensor Manager::rewardTensor() const
{
    return impl_->exportStateTensor(
        ExportID::Reward, TensorElementType::Float32,
        {impl_->cfg.numWorlds * impl_->maxAgentsPerWorld, 1});
}

Tensor Manager::agentTypeTensor() const
{
    return impl_->exportStateTensor(
        ExportID::AgentType, TensorElementType::Int32,
        {impl_->cfg.numWorlds * impl_->maxAgentsPerWorld, 1});
}

Tensor Manager::agentMaskTensor() const
{
    return impl_->exportStateTensor(
        ExportID::AgentMask, TensorElementType::Float32,
        {impl_->cfg.numWorlds * impl_->maxAgentsPerWorld, 1});
}


madrona::py::Tensor Manager::agentDataTensor() const
{
    return impl_->exportStateTensor(
        ExportID::AgentObsData, TensorElementType::Float32,
        {
            impl_->cfg.numWorlds * impl_->maxAgentsPerWorld,
            consts::maxAgents - 1,
            sizeof(AgentObservation) / sizeof(float),
        });
}

madrona::py::Tensor Manager::boxDataTensor() const
{
    return impl_->exportStateTensor(
        ExportID::BoxObsData, TensorElementType::Float32,
        {
            impl_->cfg.numWorlds * impl_->maxAgentsPerWorld,
            consts::maxBoxes,
            sizeof(BoxObservation) / sizeof(float),
        });
}

madrona::py::Tensor Manager::rampDataTensor() const
{
    return impl_->exportStateTensor(
        ExportID::RampObsData, TensorElementType::Float32,
        {
            impl_->cfg.numWorlds * impl_->maxAgentsPerWorld,
            consts::maxRamps,
            sizeof(RampObservation) / sizeof(float),
        });
}

madrona::py::Tensor Manager::visibleAgentsMaskTensor() const
{
    return impl_->exportStateTensor(
        ExportID::AgentVisMasks, TensorElementType::Float32,
        {
            impl_->cfg.numWorlds * impl_->maxAgentsPerWorld,
            consts::maxAgents - 1,
            1,
        });
}

madrona::py::Tensor Manager::visibleBoxesMaskTensor() const
{
    return impl_->exportStateTensor(
        ExportID::BoxVisMasks, TensorElementType::Float32,
        {
            impl_->cfg.numWorlds * impl_->maxAgentsPerWorld,
            consts::maxBoxes,
            1,
        });
}

madrona::py::Tensor Manager::visibleRampsMaskTensor() const
{
    return impl_->exportStateTensor(
        ExportID::RampVisMasks, TensorElementType::Float32,
        {
            impl_->cfg.numWorlds * impl_->maxAgentsPerWorld,
            consts::maxRamps,
            1,
        });
}

madrona::py::Tensor Manager::lidarTensor() const
{
    return impl_->exportStateTensor(
        ExportID::Lidar, TensorElementType::Float32,
        {
            impl_->cfg.numWorlds * impl_->maxAgentsPerWorld,
            30,
        });
}

madrona::py::Tensor Manager::seedTensor() const
{
    return impl_->exportStateTensor(
        ExportID::Seed, TensorElementType::Int32,
        {
            impl_->cfg.numWorlds * impl_->maxAgentsPerWorld,
            sizeof(Seed) / sizeof(int32_t),
        });
}

madrona::py::Tensor Manager::globalPositionsTensor() const
{
    return impl_->exportStateTensor(
        ExportID::GlobalDebugPositions, TensorElementType::Float32,
        {
            impl_->cfg.numWorlds,
            consts::maxBoxes + consts::maxRamps +
                consts::maxAgents,
            2,
        });
}

Tensor Manager::depthTensor() const
{
    const float *depth_ptr = impl_->renderMgr->batchRendererDepthOut();

    return Tensor((void *)depth_ptr, TensorElementType::Float32, {
        impl_->cfg.numWorlds * impl_->maxAgentsPerWorld,
        impl_->cfg.batchRenderViewHeight,
        impl_->cfg.batchRenderViewWidth,
        1,
    }, impl_->cfg.gpuID);
}

Tensor Manager::rgbTensor() const
{
    const uint8_t *rgb_ptr = impl_->renderMgr->batchRendererRGBOut();

    return Tensor((void *)rgb_ptr, TensorElementType::UInt8, {
        impl_->cfg.numWorlds * impl_->maxAgentsPerWorld,
        impl_->cfg.batchRenderViewHeight,
        impl_->cfg.batchRenderViewWidth,
        4,
    }, impl_->cfg.gpuID);
}

Tensor Manager::raycastTensor() const
{
    uint32_t pixels_per_view = impl_->raycastOutputResolution *
        impl_->raycastOutputResolution;
    return impl_->exportStateTensor(ExportID::Raycast,
                               TensorElementType::UInt8,
                               {
                                   impl_->cfg.numWorlds*consts::maxAgents,
                                   pixels_per_view * 3,
                               });
}

void Manager::triggerReset(CountT world_idx, CountT level_idx)
{
    WorldReset reset {
        (int32_t)level_idx,
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

render::RenderManager & Manager::getRenderManager()
{
    return *impl_->renderMgr;
}


void Manager::bpsDumpRGB() const
{
    static int num_frames = 0;
    float *gpu_ptr = impl_->bps3DState->renderer.getDepthPointer();

    uint32_t img_width = impl_->cfg.batchRenderViewWidth;
    uint32_t img_height = impl_->cfg.batchRenderViewHeight;

    uint64_t num_bytes = (uint64_t)impl_->cfg.numWorlds *
        (uint64_t)img_width * (uint64_t)img_height * sizeof(float);

    float *cpu_ptr = (float *)cu::allocReadback(num_bytes);
    cudaMemcpy(cpu_ptr, gpu_ptr, num_bytes, cudaMemcpyDeviceToHost);

    uint8_t *out_ptr = (uint8_t *)malloc(
        (uint64_t)img_width * (uint64_t)img_height * 4);

    for (uint32_t i = 0; i < impl_->cfg.numWorlds; i++) {
        float *src = cpu_ptr + (uint64_t)i * 
            (uint64_t)img_width * (uint64_t)img_height;

        for (uint32_t y = 0; y < img_height; y++) {
            for (int x = 0; x < img_height; x++) {
                float depth = src[y * img_width + x];
                depth = std::clamp(depth, 0.f, 1.f);
#if 0
                if (x == 0 && y == 0) {
                    printf("%f\n", depth);
                }
#endif

                uint8_t *out_base = out_ptr + y * img_width * 4 + x * 4;
                for (int c = 0; c < 3; c++) {
                    out_base[c] = uint8_t(depth * 255.f);
                }
                out_base[3] = 255;
            }
        }

        stbi_write_bmp((std::string("/tmp/t/img_") + 
                        std::to_string(i) + "_" + 
                        std::to_string(num_frames)).c_str(),
            impl_->cfg.batchRenderViewWidth,
            impl_->cfg.batchRenderViewHeight,
            4,
            out_ptr);
    }

    free(out_ptr);
    cudaFree(cpu_ptr);

    num_frames += 1;
}

}
