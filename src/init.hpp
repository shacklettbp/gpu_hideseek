#pragma once

#include <madrona/physics.hpp>

namespace GPUHideSeek {

struct EpisodeManager {
    std::atomic_uint32_t curEpisode;
};

struct WorldInit {
    EpisodeManager *episodeMgr;
    madrona::phys::ObjectManager *rigidBodyObjMgr;
    uint32_t minEntitiesPerWorld;
    uint32_t maxEntitiesPerWorld;
};

}
