#pragma once 

#include "sim.hpp"

namespace GPUHideSeek {

inline Entity makeDynObject(
    Engine &ctx,
    Vector3 pos,
    Quat rot,
    SimObject obj_id,
    madrona::phys::ResponseType response_type = ResponseType::Dynamic,
    OwnerTeam owner_team = OwnerTeam::None,
    Diag3x3 scale = {1, 1, 1});

CountT populateStaticGeometry(Engine &ctx,
                              RNG &rng,
                              madrona::math::Vector2 level_scale);

}

#include "geo_gen.inl"
