#pragma once

#include <madrona/taskgraph.hpp>
#include <madrona/math.hpp>
#include <madrona/custom_context.hpp>
#include <madrona/components.hpp>
#include <madrona/physics.hpp>
#include <madrona/render/ecs.hpp>

#include "init.hpp"
#include "rng.hpp"

namespace GPUHideSeek {

using madrona::Entity;
using madrona::CountT;
using madrona::base::Position;
using madrona::base::Rotation;
using madrona::base::Scale;
using madrona::base::ObjectID;
using madrona::phys::Velocity;
using madrona::phys::ResponseType;
using madrona::phys::ExternalForce;
using madrona::phys::ExternalTorque;

namespace consts {

static inline constexpr int32_t maxBoxes = 9;
static inline constexpr int32_t maxRamps = 2;
static inline constexpr int32_t maxAgents = 6;

}

enum class ExportID : uint32_t {
    Reset,
    PrepCounter,
    Action,
    AgentType,
    AgentMask,
    AgentObsData,
    BoxObsData,
    RampObsData,
    AgentVisMasks,
    BoxVisMasks,
    RampVisMasks,
    Lidar,
    Seed,
    Reward,
    Done,
    GlobalDebugPositions,
    NumExports,
};

enum class TaskGraphID : uint32_t {
    Step,
    NumTaskGraphs,
};

enum class SimObject : uint32_t {
    Sphere,
    Plane,
    Cube,
    Wall,
    Agent,
    Ramp,
    Box,
    NumObjects,
};

struct Config {
    bool autoReset;
    uint32_t maxAgentsPerWorld;
    madrona::phys::ObjectManager *rigidBodyObjMgr;
    const madrona::render::RenderECSBridge *renderBridge;
};

class Engine;

struct WorldReset {
    int32_t resetLevel;
    int32_t numHiders;
    int32_t numSeekers;
};

struct AgentPrepCounter {
    int32_t numPrepStepsLeft;
};

enum class OwnerTeam : uint32_t {
    None,
    Seeker,
    Hider,
    Unownable,
};

struct GrabData {
    Entity constraintEntity;
};

enum class AgentType : uint32_t {
    Seeker = 0,
    Hider = 1,
};

struct DynamicObject : public madrona::Archetype<
    Position, 
    Rotation,
    Scale,
    Velocity,
    ObjectID,
    ResponseType,
    madrona::phys::solver::SubstepPrevState,
    madrona::phys::solver::PreSolvePositional,
    madrona::phys::solver::PreSolveVelocity,
    ExternalForce,
    ExternalTorque,
    madrona::phys::broadphase::LeafID,
    OwnerTeam,
    madrona::render::Renderable
> {};

struct Action {
    int32_t x;
    int32_t y;
    int32_t r;
    int32_t g;
    int32_t l;
};

struct SimEntity {
    Entity e;
};

struct AgentActiveMask {
    float mask;
};

struct GlobalDebugPositions {
    madrona::math::Vector2 boxPositions[consts::maxBoxes];
    madrona::math::Vector2 rampPositions[consts::maxRamps];
    madrona::math::Vector2 agentPositions[consts::maxAgents];
};

struct AgentObservation {
    madrona::math::Vector2 pos;
    madrona::math::Vector2 vel;
};

struct BoxObservation {
    madrona::math::Vector2 pos;
    madrona::math::Vector2 vel;
    madrona::math::Vector2 boxSize;
    float boxRotation;
};

struct RampObservation {
    madrona::math::Vector2 pos;
    madrona::math::Vector2 vel;
    float rampRotation;
};

struct RelativeAgentObservations {
    AgentObservation obs[consts::maxAgents - 1];
};

struct RelativeBoxObservations {
    BoxObservation obs[consts::maxBoxes];
};

struct RelativeRampObservations {
    RampObservation obs[consts::maxRamps];
};

struct AgentVisibilityMasks {
    float visible[consts::maxAgents - 1];
};

struct BoxVisibilityMasks {
    float visible[consts::maxBoxes];
};

struct RampVisibilityMasks {
    float visible[consts::maxRamps];
};

struct Lidar {
    float depth[30];
};

struct Seed {
    int32_t seed;
};

struct Reward {
    float v;
};

struct Done {
    int32_t v;
};

static_assert(sizeof(Action) == 5 * sizeof(int32_t));

struct AgentInterface : public madrona::Archetype<
    Position,
    Rotation,
    SimEntity,
    AgentPrepCounter,
    Action,
    AgentType,
    AgentActiveMask,
    RelativeAgentObservations,
    RelativeBoxObservations,
    RelativeRampObservations,
    AgentVisibilityMasks,
    BoxVisibilityMasks,
    RampVisibilityMasks,
    Lidar,
    Seed,
    Reward,
    Done,
    madrona::render::RenderCamera
> {};

struct DynAgent : public madrona::Archetype<
    Position, 
    Rotation,
    Scale,
    Velocity,
    ObjectID,
    ResponseType,
    madrona::phys::solver::SubstepPrevState,
    madrona::phys::solver::PreSolvePositional,
    madrona::phys::solver::PreSolveVelocity,
    ExternalForce,
    ExternalTorque,
    madrona::phys::broadphase::LeafID,
    OwnerTeam,
    GrabData,
    madrona::render::Renderable
> {};

struct Sim : public madrona::WorldBase {
    static void registerTypes(madrona::ECSRegistry &registry,
                              const Config &cfg);

    static void setupTasks(madrona::TaskGraphManager &taskgraph_mgr,
                           const Config &cfg);

    Sim(Engine &ctx,
        const Config &cfg,
        const WorldInit &init);

    EpisodeManager *episodeMgr;
    RNG rng;

    Entity agentInterfaces[consts::maxAgents];

    Entity hiders[3];
    int32_t numHiders;
    Entity seekers[3];
    int32_t numSeekers;

    Entity *obstacles;
    int32_t numObstacles;
    Entity boxes[consts::maxBoxes];
    madrona::math::Vector2 boxSizes[consts::maxBoxes];
    float boxRotations[consts::maxBoxes];
    Entity ramps[consts::maxRamps];
    float rampRotations[consts::maxRamps];
    CountT numActiveBoxes;
    CountT numActiveRamps;
    CountT numActiveAgents;

    CountT curEpisodeStep;
    CountT minEpisodeEntities;
    CountT maxEpisodeEntities;

    uint32_t curEpisodeSeed;
    bool enableRender;
    bool autoReset;
    uint32_t maxAgentsPerWorld;

    madrona::AtomicFloat hiderTeamReward {0};
};

class Engine : public ::madrona::CustomContext<Engine, Sim> {
    using CustomContext::CustomContext;

public:
    // These are convenience helpers for creating renderable
    // entities when rendering isn't necessarily enabled
    template <typename ArchetypeT>
    inline madrona::Entity makeRenderableEntity();
    inline void destroyRenderableEntity(Entity e);
};

}

#include "sim.inl"
