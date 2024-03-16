#pragma once

#include <memory>

#include <madrona/py/utils.hpp>
#include <madrona/exec_mode.hpp>

#include <madrona/render/render_mgr.hpp>

#include "sim_flags.hpp"

namespace GPUHideSeek {

class Manager {
public:
    struct Config {
        madrona::ExecMode execMode;
        int gpuID;
        uint32_t numWorlds;
        SimFlags simFlags;
        uint32_t randSeed;
        uint32_t minHiders;
        uint32_t maxHiders;
        uint32_t minSeekers;
        uint32_t maxSeekers;
        bool enableBatchRenderer;
        uint32_t batchRenderViewWidth = 64;
        uint32_t batchRenderViewHeight = 64;
        madrona::render::APIBackend *extRenderAPI = nullptr;
        madrona::render::GPUDevice *extRenderDev = nullptr;
    };

    Manager(const Config &cfg);
    ~Manager();

    void init();
    void step();

    madrona::py::Tensor resetTensor() const;
    madrona::py::Tensor doneTensor() const;
    madrona::py::Tensor prepCounterTensor() const;
    madrona::py::Tensor actionTensor() const;
    madrona::py::Tensor rewardTensor() const;
    madrona::py::Tensor agentTypeTensor() const;
    madrona::py::Tensor agentMaskTensor() const;
    madrona::py::Tensor agentDataTensor() const;
    madrona::py::Tensor boxDataTensor() const;
    madrona::py::Tensor rampDataTensor() const;
    madrona::py::Tensor visibleAgentsMaskTensor() const;
    madrona::py::Tensor visibleBoxesMaskTensor() const;
    madrona::py::Tensor visibleRampsMaskTensor() const;
    madrona::py::Tensor globalPositionsTensor() const;
    madrona::py::Tensor lidarTensor() const;
    madrona::py::Tensor seedTensor() const;

    madrona::py::Tensor depthTensor() const;
    madrona::py::Tensor rgbTensor() const;

    void triggerReset(madrona::CountT world_idx,
                      madrona::CountT level_idx);
    void setAction(madrona::CountT agent_idx,
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
