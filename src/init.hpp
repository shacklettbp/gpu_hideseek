#pragma once

#include <madrona/physics.hpp>
#include <madrona/render/ecs.hpp>

namespace GPUHideSeek {

struct EpisodeManager {
    madrona::AtomicU32 curEpisode;
};

struct WorldInit {
    EpisodeManager *episodeMgr;
    float *rewardBuffer;
    uint8_t *doneBuffer;
    uint32_t minEntitiesPerWorld;
    uint32_t maxEntitiesPerWorld;
};

}
