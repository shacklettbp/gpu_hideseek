from madrona_learn import (
    ActorCritic, DiscreteActor, Critic, 
    BackboneShared, BackboneSeparate,
    BackboneEncoder, RecurrentBackboneEncoder,
)

from madrona_learn.models import (
    MLP, LinearLayerDiscreteActor, LinearLayerCritic,
)

from madrona_learn.rnn import LSTM, FastLSTM

import math
import torch
import torch.nn as nn

def setup_obs(sim):
    N = sim.reset_tensor().to_torch().shape[0]

    prep_counter = sim.prep_counter_tensor().to_torch()[0:N * 5, ...]
    agent_type = sim.agent_type_tensor().to_torch()[0:N * 5, ...]
    agent_data = sim.agent_data_tensor().to_torch()[0:N * 5, ...]
    box_data = sim.box_data_tensor().to_torch()[0:N * 5, ...]
    ramp_data = sim.ramp_data_tensor().to_torch()[0:N * 5, ...]
    visible_agents_mask = sim.visible_agents_mask_tensor().to_torch()[0:N * 5, ...]
    visible_boxes_mask = sim.visible_boxes_mask_tensor().to_torch()[0:N * 5, ...]
    visible_ramps_mask = sim.visible_ramps_mask_tensor().to_torch()[0:N * 5, ...]
    lidar_tensor = sim.lidar_tensor().to_torch()[0:N * 5, ...]

    # Add in an agent ID tensor
    id_tensor = torch.arange(5).float()

    id_tensor = id_tensor.to(device=prep_counter.device)
    id_tensor = id_tensor.view(1, 5).expand(prep_counter.shape[0] // 5, 5).reshape(
        prep_counter.shape[0], 1)

    obs_tensors = [
        prep_counter,
        agent_type,
        agent_data,
        box_data,
        ramp_data,
        lidar_tensor,
        id_tensor,
    ]

    num_obs_features = 0
    for tensor in obs_tensors:
        num_obs_features += math.prod(tensor.shape[1:])

    obs_tensors += [
        visible_agents_mask,
        visible_boxes_mask,
        visible_ramps_mask,
    ]

    return obs_tensors, num_obs_features

def flatten(tensor):
    return tensor.view(tensor.shape[0], -1)


def process_obs(prep_counter,
                agent_type,
                agent_data,
                box_data,
                ramp_data,
                lidar,
                ids,
                visible_agents_mask,
                visible_boxes_mask,
                visible_ramps_mask,
            ):
    assert(not torch.isnan(prep_counter).any())
    assert(not torch.isinf(prep_counter).any())

    assert(not torch.isnan(agent_type).any())
    assert(not torch.isinf(agent_type).any())

    assert(not torch.isnan(agent_data).any())
    assert(not torch.isinf(agent_data).any())

    assert(not torch.isnan(box_data).any())
    assert(not torch.isinf(box_data).any())

    assert(not torch.isnan(ramp_data).any())
    assert(not torch.isinf(ramp_data).any())

    assert(not torch.isnan(lidar).any())
    assert(not torch.isinf(lidar).any())

    assert(not torch.isnan(visible_agents_mask).any())
    assert(not torch.isinf(visible_agents_mask).any())

    assert(not torch.isnan(visible_boxes_mask).any())
    assert(not torch.isinf(visible_boxes_mask).any())

    assert(not torch.isnan(visible_ramps_mask).any())
    assert(not torch.isinf(visible_ramps_mask).any())


    common = torch.cat([
            flatten(prep_counter.float() / 200),
            flatten(agent_type),
            ids,
        ], dim=1)

    not_common = [
            lidar,
            agent_data,
            box_data,
            ramp_data,
            visible_agents_mask,
            visible_boxes_mask,
            visible_ramps_mask,
        ]

    return (common, not_common)


class ActorEncoderNet(nn.Module):
    def __init__(self, num_obs_features, num_channels):
        super().__init__()

        self.net = MLP(
            input_dim = num_obs_features,
            num_channels = num_channels,
            num_layers = 2,
        )

        self.lidar_conv = nn.Conv1D(in_channels=30, out_channels=30,
                                    kernel_size=3, padding=1,
                                    padding_mode='circular')


    def forward(self, x):
        common, (lidar, agent_data, box_data, ramp_data, visible_agents_mask, visible_boxes_mask, visible_ramps_mask) = x

        with torch.no_grad():
            masked_agents = agent_data * visible_agents_mask
            masked_boxes = box_data * visible_boxes_mask
            masked_ramps = ramp_data * visible_ramps_mask

        lidar_processed = self.lidar_conv(lidar)

        combined = torch.cat([
                common,
                flatten(lidar_processed),
                flatten(masked_agents),
                flatten(masked_boxes),
                flatten(masked_ramps),
            ], dim=1)

        return self.net(combined)


class CriticEncoderNet(nn.Module):
    def __init__(self, num_obs_features, num_channels):
        super().__init__()

        self.net = MLP(
            input_dim = num_obs_features,
            num_channels = num_channels,
            num_layers = 2,
        )

        self.lidar_conv = nn.Conv1D(in_channels=30, out_channels=30,
                                    kernel_size=3, padding=1,
                                    padding_mode='circular')

    def forward(self, x):
        common, (lidar, agent_data, box_data, ramp_data, visible_agents_mask, visible_boxes_mask, visible_ramps_mask) = x

        lidar_processed = self.lidar_conv(lidar)

        combined = torch.cat([
                common,
                flatten(lidar_processed),
                flatten(agent_data),
                flatten(box_data),
                flatten(ramp_data),
            ], dim=1)

        return self.net(combined)


def make_policy(num_obs_features, num_channels):
    actor_encoder = RecurrentBackboneEncoder(
        net = ActorEncoderNet(num_obs_features, num_channels),
        rnn = LSTM(
            in_channels = num_channels,
            hidden_channels = num_channels,
            num_layers = 1,
        ),
    )

    critic_encoder = RecurrentBackboneEncoder(
        net = CriticEncoderNet(num_obs_features, num_channels),
        rnn = LSTM(
            in_channels = num_channels,
            hidden_channels = num_channels,
            num_layers = 1,
        ),
    )

    backbone = BackboneSeparate(
        process_obs = process_obs,
        actor_encoder = actor_encoder,
        critic_encoder = critic_encoder,
    )

    return ActorCritic(
        backbone = backbone,
        actor = LinearLayerDiscreteActor(
            [11, 11, 11, 2, 2],
            num_channels,
        ),
        critic = LinearLayerCritic(num_channels),
    )
