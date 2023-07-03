#pragma once

#include <madrona/physics.hpp>
#include <madrona/render/mw.hpp>
#include <madrona/viz/system.hpp>

namespace GPUHideSeek {

struct EpisodeManager {
    madrona::AtomicU32 curEpisode;
};

struct WorldInit {
    EpisodeManager *episodeMgr;
    float *rewardBuffer;
    uint8_t *doneBuffer;
    madrona::phys::ObjectManager *rigidBodyObjMgr;
    uint32_t minEntitiesPerWorld;
    uint32_t maxEntitiesPerWorld;
    const madrona::viz::VizECSBridge *vizBridge;
    const madrona::render::BatchRendererECSBridge *batchRenderBridge;
};

}
