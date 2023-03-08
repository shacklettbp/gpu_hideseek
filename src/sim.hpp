#pragma once

#include <madrona/taskgraph.hpp>
#include <madrona/math.hpp>
#include <madrona/custom_context.hpp>
#include <madrona/components.hpp>
#include <madrona/physics.hpp>
#include <madrona/mw_render.hpp>

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

class Engine;

struct WorldReset {
    int32_t resetLevel;
    int32_t numHiders;
    int32_t numSeekers;
};

struct WorldDone {
    int32_t done;
};

struct PrepPhaseCounter {
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
    Camera = 2,
};

struct PhysicsObject {
    Position pos;
    Rotation rot;
    Scale scale;
    Velocity vel;
    ObjectID objID;
    ResponseType respType;
    madrona::phys::solver::SubstepPrevState substepPrevState;
    madrona::phys::solver::PreSolvePositional preSolvePositional;
    madrona::phys::solver::PreSolveVelocity preSolveVelocity;
    ExternalForce extForce;
    ExternalTorque extTorque;
    madrona::phys::broadphase::LeafID leafID;
};

struct Obstacle : public PhysicsObject {
    OwnerTeam ownerTeam;
};

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

struct Reward {
    float reward;
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

static_assert(sizeof(Action) == 5 * sizeof(int32_t));

struct AgentInterfaces {
    SimEntity *simEntities;
    Action *actions;
    Reward *rewards;
    AgentType *agentTypes;
    AgentActiveMask *agentActiveMasks;
    RelativeAgentObservations *relAgentObs;
    RelativeBoxObservations *relBoxObs;
    RelativeRampObservations *relRampObs;
    AgentVisibilityMasks *agentVisMasks;
    BoxVisibilityMasks *boxVisMasks;
    RampVisibilityMasks *rampVisMasks;
    Lidar *lidar;
};

struct DynAgent : PhysicsObject {
    OwnerTeam team;
    GrabData grabData;
    madrona::render::ViewSettings viewSettings;
};

struct Config {
    bool enableRender;
};

struct Sim : public madrona::WorldBase {
    static void registerTypes(madrona::ECSRegistry &registry,
                              const Config &cfg);

    static void setupTasks(madrona::TaskGraph::Builder &builder,
                           const Config &cfg);

    Sim(Engine &ctx,
        const Config &cfg,
        const WorldInit &init);

    EpisodeManager *episodeMgr;
    RNG rng;

    Entity *obstacles;
    CountT numObstacles;

    int32_t hiders[3];
    CountT numHiders;
    int32_t seekers[3];
    CountT numSeekers;

    Obstacle boxes[consts::maxBoxes];
    madrona::math::Vector2 boxSizes[consts::maxBoxes];
    float boxRotations[consts::maxBoxes];
    Obstacle ramps[consts::maxRamps];
    float rampRotations[consts::maxRamps];

    AgentInterface agentInterfaces;
    CountT numActiveBoxes;
    CountT numActiveRamps;
    CountT numActiveAgents;

    CountT curEpisodeStep;
    CountT minEpisodeEntities;
    CountT maxEpisodeEntities;

    bool enableRender;

    madrona::AtomicFloat hiderTeamReward {0};
};

class Engine : public ::madrona::CustomContext<Engine, Sim> {
    using CustomContext::CustomContext;
};

}
