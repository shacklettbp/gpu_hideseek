#pragma once
#ifdef gpu_hideseek_mgr_EXPORTS
#define MGR_EXPORT MADRONA_EXPORT
#else
#define MGR_EXPORT MADRONA_IMPORT
#endif

#include <memory>

#include <madrona/py/utils.hpp>
#include <madrona/exec_mode.hpp>

#include <madrona/render/render_mgr.hpp>

namespace GPUHideSeek {

class Manager {
public:
    struct Config {
        madrona::ExecMode execMode;
        int gpuID;
        uint32_t numWorlds;
        uint32_t renderWidth;
        uint32_t renderHeight;
        bool autoReset;
        bool enableBatchRenderer;
        uint32_t batchRenderViewWidth = 64;
        uint32_t batchRenderViewHeight = 64;
        madrona::render::APIBackend *extRenderAPI = nullptr;
        madrona::render::GPUDevice *extRenderDev = nullptr;
    };

    MGR_EXPORT Manager(const Config &cfg);
    MGR_EXPORT ~Manager();

    MGR_EXPORT void step();

    MGR_EXPORT madrona::py::Tensor resetTensor() const;
    MGR_EXPORT madrona::py::Tensor doneTensor() const;
    MGR_EXPORT madrona::py::Tensor prepCounterTensor() const;
    MGR_EXPORT madrona::py::Tensor actionTensor() const;
    MGR_EXPORT madrona::py::Tensor rewardTensor() const;
    MGR_EXPORT madrona::py::Tensor agentTypeTensor() const;
    MGR_EXPORT madrona::py::Tensor agentMaskTensor() const;
    MGR_EXPORT madrona::py::Tensor agentDataTensor() const;
    MGR_EXPORT madrona::py::Tensor boxDataTensor() const;
    MGR_EXPORT madrona::py::Tensor rampDataTensor() const;
    MGR_EXPORT madrona::py::Tensor visibleAgentsMaskTensor() const;
    MGR_EXPORT madrona::py::Tensor visibleBoxesMaskTensor() const;
    MGR_EXPORT madrona::py::Tensor visibleRampsMaskTensor() const;
    MGR_EXPORT madrona::py::Tensor globalPositionsTensor() const;
    MGR_EXPORT madrona::py::Tensor lidarTensor() const;
    MGR_EXPORT madrona::py::Tensor seedTensor() const;

    MGR_EXPORT madrona::py::Tensor depthTensor() const;
    MGR_EXPORT madrona::py::Tensor rgbTensor() const;

    MGR_EXPORT void triggerReset(madrona::CountT world_idx,
                                 madrona::CountT level_idx,
                                 madrona::CountT num_hiders,
                                 madrona::CountT num_seekers);
    MGR_EXPORT void setAction(madrona::CountT agent_idx,
                              int32_t x, int32_t y, int32_t r,
                              bool g, bool l);

    madrona::render::RenderManager & getRenderManager();

private:
    struct Impl;
    struct CPUImpl;
    struct CUDAImpl;

    Impl *impl_;
};

}
