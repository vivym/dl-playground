from typing import Optional, List, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.fc = nn.Linear(in_channels, out_channels)
        self.norm = nn.LayerNorm(out_channels)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ):
        x = x[mask]
        x = self.fc(x)
        x = self.norm(x)
        x = F.leaky_relu(x)

        out = torch.empty(
            mask.shape + (self.out_channels,), dtype=x.dtype, device=x.device
        )
        out[mask] = x

        pool = out.clone()
        pool[~mask] = -float("inf")
        pool, _ = torch.max(pool, dim=2, keepdim=True)
        pool = pool.expand_as(out)

        return torch.cat([out, pool], dim=-1)


class SubGraph(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.mlp1 = MLP(in_channels, out_channels)
        self.mlp2 = MLP(2 * out_channels, out_channels)
        self.mlp3 = MLP(2 * out_channels, out_channels)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ):
        x = self.mlp1(x, mask)
        x = self.mlp2(x, mask)
        x = self.mlp3(x, mask)

        x = x[:, :, 0, -self.out_channels:]
        x[~torch.any(mask, dim=2), :] = 0

        return x


class GNN(nn.Module):
    def __init__(
        self,
        num_channels: int,
        num_heads: int,
    ):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim=num_channels, num_heads=num_heads
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        query = x / (torch.sum(x ** 2, dim=-1, keepdim=True) ** 0.5 + 1e-6)
        query = query.transpose(0, 1)
        key = query
        value = x.transpose(0, 1)

        x, _ = self.attn(
            query=query, key=key, value=value, key_padding_mask=~mask
        )
        x = x.transpose(0, 1)

        return x


class Decoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_modes: int,
        timesteps: int,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.num_modes = num_modes
        self.timesteps = timesteps

        self.fc1 = nn.Linear(in_channels, 4096)
        self.fc2 = nn.Linear(4096, 512)

        self.reg_head = nn.Linear(512, num_modes * timesteps * 2)
        self.cls_head = nn.Linear(512, num_modes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        pos = self.reg_head(x)
        pos = pos.view(-1, self.num_modes, self.timesteps, 2)

        prob = self.cls_head(x)
        prob = prob.sum(0)
        prob = prob - prob.max()
        prob = F.softmax(prob, dim=0)

        return pos, prob


class VectorNet(pl.LightningModule):
    def __init__(
        self,
        agent_in_channels: int,
        roadmap_in_channels: int,
        num_channels: int,
        learning_rate: float = 1e-3,
        max_epochs: int = 60,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.agent_in_channels = agent_in_channels
        self.roadmap_in_channels = roadmap_in_channels
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs

        self.agent_subgraph = SubGraph(agent_in_channels, num_channels)
        self.roadmap_subgraph = SubGraph(roadmap_in_channels, num_channels)
        self.global_graph = GNN(num_channels, num_heads=1)
        self.decoder = Decoder(2 * num_channels + 5, num_modes=3, timesteps=80)

    def forward(
        self,
        agents_polylines: torch.Tensor,
        agents_polylines_mask: torch.Tensor,
        roadgraph_polylines: torch.Tensor,
        roadgraph_polylines_mask: torch.Tensor,
        objects_of_interest: torch.Tensor,
        agents_motion: torch.Tensor,
    ):
        batch_size = agents_polylines.shape[0]

        # agents_polylines: batch_size, num_agents, timestamps, channels
        # roadgraph_polylines: batch_size, num_roads, road_length, channels
        # print("agents_polylines", agents_polylines.shape)
        # print("agents_polylines_mask", agents_polylines_mask.shape)
        # print("roadgraph_polylines", roadgraph_polylines.shape)
        # print("roadgraph_polylines_mask", roadgraph_polylines_mask.shape)
        # print("objects_of_interest", objects_of_interest.shape)
        # print("agents_motion", agents_motion.shape)

        agent_polyline_features = self.agent_subgraph(
            agents_polylines[:, :, :, :self.agent_in_channels],
            agents_polylines_mask,
        )

        # print("agent_polyline_features", agent_polyline_features.shape)

        roadgraph_polyline_features = self.roadmap_subgraph(
            roadgraph_polylines[:, :, :, :self.roadmap_in_channels],
            roadgraph_polylines_mask,
        )

        agent_feature_mask = agents_polylines_mask.any(2)
        roadgraph_feature_mask = roadgraph_polylines_mask.any(2)

        graph_in_features = torch.cat([
            agent_polyline_features, roadgraph_polyline_features
        ], dim=1)
        graph_mask = torch.cat([agent_feature_mask, roadgraph_feature_mask], dim=1)
        graph_out_features = self.global_graph(graph_in_features, graph_mask)
        graph_features = torch.cat([graph_out_features, graph_in_features], dim=2)

        trajs_list, probs_list, agent_inds_list = [], [], []
        for i in range(batch_size):
            agent_inds = torch.nonzero((objects_of_interest == 1)[i], as_tuple=True)[0]

            target_features = torch.cat([
                graph_features[i, agent_inds, :],
                agents_polylines[i, agent_inds, 0, :2],
                agents_motion[i, agent_inds, :],
            ], dim=1)

            decoded_trajs, decoded_probs = self.decoder(target_features)
            trajs_list.append(decoded_trajs)
            probs_list.append(decoded_probs)
            agent_inds_list.append(agent_inds)

        return trajs_list, probs_list, agent_inds_list

    def loss_fn(self, trajs, probs, targets):
        num_timesteps = trajs.shape[2]

        targets = targets[:, None, :, :]
        sq_dif = torch.square(trajs - targets)
        l2_per_timestep = torch.sqrt(torch.sum(sq_dif, dim=3))
        ade_per_actor_per_mode = torch.sum(l2_per_timestep, dim=2) / num_timesteps
        ade_per_mode = torch.sum(ade_per_actor_per_mode, 0)

        best_mode = torch.argmin(ade_per_mode, dim=0)
        min_ade = torch.index_select(ade_per_mode, dim=0, index=best_mode)
        min_ade_prob = torch.index_select(probs, dim=0, index=best_mode)
        min_ade_ce = -torch.log(min_ade_prob + 1e-5)

        return min_ade, min_ade_ce

    def _training_and_validation_step(self, batch, batch_idx: int):
        (
            agents_polylines, agents_polylines_mask,
            roadgraph_polylines, roadgraph_polylines_mask,
            targets, targets_mask,
            tracks_to_predict, objects_of_interest,
            agents_motion,
        ) = batch

        trajs_list, probs_list, agent_inds_list = self.forward(
            agents_polylines=agents_polylines,
            agents_polylines_mask=agents_polylines_mask,
            roadgraph_polylines=roadgraph_polylines,
            roadgraph_polylines_mask=roadgraph_polylines_mask,
            objects_of_interest=objects_of_interest,
            agents_motion=agents_motion,
        )

        loss_ade, loss_ade_ce = 0, 0
        num_tracks_to_predict = 0
        for i, (trajs, probs, agent_inds) in enumerate(
            zip(trajs_list, probs_list, agent_inds_list)
        ):
            num_tracks_to_predict += agent_inds.shape[0]

            targets_i = targets[i, agent_inds, :, :2]
            loss_ade_i, loss_ade_ce_i = self.loss_fn(trajs, probs, targets_i)
            loss_ade += loss_ade_i
            loss_ade_ce += loss_ade_ce_i

        loss_ade = loss_ade / num_tracks_to_predict
        loss_ade_ce = loss_ade_ce / num_tracks_to_predict

        loss = loss_ade + loss_ade_ce

        return loss, loss_ade, loss_ade_ce

    def training_step(self, batch, batch_idx: int):
        loss, loss_ade, loss_ade_ce = self._training_and_validation_step(batch, batch)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            "train/loss_ade", loss_ade,
            on_step=True, on_epoch=True, prog_bar=True,
        )
        self.log(
            "train/loss_ade_ce", loss_ade_ce,
            on_step=True, on_epoch=True, prog_bar=True,
        )

        return loss

    def validation_step(self, batch, batch_idx: int):
        loss, loss_ade, loss_ade_ce = self._training_and_validation_step(batch, batch)

        self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            "val/loss_ade", loss_ade,
            on_step=True, on_epoch=True, prog_bar=True,
        )
        self.log(
            "val/loss_ade_ce", loss_ade_ce,
            on_step=True, on_epoch=True, prog_bar=True,
        )

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.learning_rate,
        )
