#include "mgr.hpp"

#include <madrona/macros.hpp>
#include <madrona/py/bindings.hpp>

#if defined(MADRONA_CLANG) || defined(MADRONA_GCC)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weverything"
#endif
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#if defined(MADRONA_CLANG) || defined(MADRONA_GCC)
#pragma GCC diagnostic pop
#endif

namespace nb = nanobind;

namespace GPUHideSeek {

NB_MODULE(gpu_hideseek, m) {
    madrona::py::setupMadronaSubmodule(m);

    nb::class_<Manager> (m, "HideAndSeekSimulator")
        .def("__init__", [](Manager *self,
                            madrona::py::PyExecMode exec_mode,
                            int64_t gpu_id,
                            int64_t num_worlds,
                            bool auto_reset,
                            uint32_t max_agents_per_world,
                            bool enable_batch_render,
                            int64_t batch_render_width,
                            int64_t batch_render_height) {
            new (self) Manager(Manager::Config {
                .execMode = exec_mode,
                .gpuID = (int)gpu_id,
                .numWorlds = (uint32_t)num_worlds,
                .autoReset = auto_reset,
                .maxAgentsPerWorld = max_agents_per_world,
                .enableBatchRenderer = enable_batch_render,
                .batchRenderViewWidth = (uint32_t)batch_render_width,
                .batchRenderViewHeight = (uint32_t)batch_render_height,
            });
        }, nb::arg("exec_mode"),
           nb::arg("gpu_id"),
           nb::arg("num_worlds"),
           nb::arg("auto_reset"),
           nb::arg("max_agents_per_world"),
           nb::arg("enable_batch_renderer") = false,
           nb::arg("batch_render_width") = 64,
           nb::arg("batch_render_height") = 64)
        .def("init", &Manager::init)
        .def("step", &Manager::step)
        .def("reset_tensor", &Manager::resetTensor)
        .def("done_tensor", &Manager::doneTensor)
        .def("prep_counter_tensor", &Manager::prepCounterTensor)
        .def("action_tensor", &Manager::actionTensor)
        .def("reward_tensor", &Manager::rewardTensor)
        .def("agent_type_tensor", &Manager::agentTypeTensor)
        .def("agent_mask_tensor", &Manager::agentMaskTensor)
        .def("agent_data_tensor", &Manager::agentDataTensor)
        .def("box_data_tensor", &Manager::boxDataTensor)
        .def("ramp_data_tensor", &Manager::rampDataTensor)
        .def("visible_agents_mask_tensor", &Manager::visibleAgentsMaskTensor)
        .def("visible_boxes_mask_tensor", &Manager::visibleBoxesMaskTensor)
        .def("visible_ramps_mask_tensor", &Manager::visibleRampsMaskTensor)
        .def("global_positions_tensor", &Manager::globalPositionsTensor)
        .def("depth_tensor", &Manager::depthTensor)
        .def("rgb_tensor", &Manager::rgbTensor)
        .def("lidar_tensor", &Manager::lidarTensor)
        .def("seed_tensor", &Manager::seedTensor)
    ;
}

}
