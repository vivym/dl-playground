from typing import Optional, List, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_scatter import scatter_max, scatter_mean

from dl_playground.data.datasets.waymo_motion import Polylines


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


class SubGraph2(nn.Module):
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


class SubGraphLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.fc = nn.Linear(in_channels, out_channels)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x: Polylines):
        x.features = self.fc(x.features)
        x.features = F.leaky_relu(x.features)

        agg_features, _ = scatter_max(
            src=x.features,
            index=x.agg_indices,
            dim=0,
        )

        x.features = torch.cat([
            x.features, agg_features[x.agg_indices]
        ], dim=-1)

        return x, agg_features


class SubGraph(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 3,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.layers = nn.ModuleList([
            SubGraphLayer(in_channels if i == 0 else 2 * out_channels, out_channels)
            for i in range(num_layers)
        ])

    def forward(self, x: Polylines) -> torch.Tensor:
        for layer in self.layers:
            x, agg_features = layer(x)
        return agg_features


class GlobalGraph(nn.Module):
    def __init__(
        self,
        num_channels: int,
        num_heads: int = 1,
        num_layers: int = 1,
        dropout: float = 0.,
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            gnn.TransformerConv(
                num_channels, num_channels,
                heads=num_heads,
                dropout=dropout,
                concat=False,
            )
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, edge_indices: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, edge_indices)

        return x


class Decoder(nn.Module):
    def __init__(
        self,
        num_channels: List[int],
        num_modes: int,
        timesteps: int,
    ):
        super().__init__()

        self.num_channels = num_channels
        self.num_modes = num_modes
        self.timesteps = timesteps

        self.layers = nn.Sequential()
        for in_channels, out_channels in zip(num_channels, num_channels[1:]):
            self.layers.append(nn.Linear(in_channels, out_channels))
            self.layers.append(nn.LeakyReLU(inplace=True))

        self.reg_head = nn.Linear(num_channels[-1], num_modes * timesteps * 2)
        self.cls_head = nn.Linear(num_channels[-1], num_modes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.layers(x)

        pred_trajs = self.reg_head(x)
        pred_trajs = pred_trajs.view(-1, self.num_modes, self.timesteps, 2)

        logits = self.cls_head(x)

        return pred_trajs, logits


class VectorNet(pl.LightningModule):
    def __init__(
        self,
        agent_in_channels: int,
        roadmap_in_channels: int,
        num_channels: int,
        num_subgraph_layers: int = 3,
        num_global_graph_layers: int = 1,
        num_global_graph_heads: int = 1,
        global_graph_dropout: float = 0.,
        learning_rate: float = 1e-3,
        max_epochs: int = 60,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.agent_in_channels = agent_in_channels
        self.roadmap_in_channels = roadmap_in_channels
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs

        # self.agent_subgraph = SubGraph(
        #     agent_in_channels, num_channels, num_layers=num_subgraph_layers
        # )
        # self.roadmap_subgraph = SubGraph(
        #     roadmap_in_channels, num_channels, num_layers=num_subgraph_layers
        # )
        self.agent_subgraph = SubGraph(
            agent_in_channels, num_channels, num_layers=num_subgraph_layers
        )
        self.roadmap_subgraph = SubGraph(
            roadmap_in_channels, num_channels, num_layers=num_subgraph_layers
        )
        self.global_graph = GlobalGraph(
            num_channels,
            num_heads=num_global_graph_heads,
            num_layers=num_global_graph_layers,
            dropout=global_graph_dropout,
        )
        # self.global_graph = GNN(num_channels, num_heads=1)
        self.decoder = Decoder([2 * num_channels + 2, 1024, 2048], num_modes=3, timesteps=80)

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

    def forward(
        self,
        agents_polylines: torch.Tensor,
        agents_polylines_mask: torch.Tensor,
        roadgraph_polylines: torch.Tensor,
        roadgraph_polylines_mask: torch.Tensor,
        objects_of_interest: torch.Tensor,
        agents_motion: torch.Tensor,
        targets: torch.Tensor,
        targets_mask: torch.Tensor,
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

        batch_size, num_agents, timestamps, channels = agents_polylines.shape
        tmp_mask = agents_polylines_mask.view(batch_size * num_agents, timestamps)
        valid_agents = tmp_mask.any(-1)
        tmp_in = agents_polylines.view(batch_size * num_agents, timestamps, channels)
        tmp_in2 = tmp_in[valid_agents]
        tmp_mask_2 = tmp_mask[valid_agents]

        tmp_in = tmp_in2[tmp_mask_2]

        agg_indices = tmp_mask_2.nonzero(as_tuple=True)[0]
        tmp_agents_polylines = Polylines(tmp_in, agg_indices)
        tmp_out = self.agent_subgraph(tmp_agents_polylines)

        tmp_out_3 = torch.zeros(batch_size * num_agents, tmp_out.shape[-1], dtype=tmp_out.dtype, device=tmp_in2.device)
        tmp_out_3[valid_agents] = tmp_out

        agent_polyline_features = tmp_out_3.view(batch_size, num_agents, tmp_out.shape[-1])

        # roadgraph_polyline_features = self.roadmap_subgraph(
        #     roadgraph_polylines[:, :, :, :self.roadmap_in_channels],
        #     roadgraph_polylines_mask,
        # )

        ################################################################
        batch_size, num_roads, timestamps, channels = roadgraph_polylines.shape
        tmp_mask = roadgraph_polylines_mask.view(batch_size * num_roads, timestamps)
        valid_roads = tmp_mask.any(-1)
        tmp_in = roadgraph_polylines.view(batch_size * num_roads, timestamps, channels)
        tmp_in2 = tmp_in[valid_roads]
        tmp_mask_2 = tmp_mask[valid_roads]

        tmp_in = tmp_in2[tmp_mask_2]

        agg_indices = tmp_mask_2.nonzero(as_tuple=True)[0]
        tmp_roads_polylines = Polylines(tmp_in, agg_indices)
        tmp_out = self.roadmap_subgraph(tmp_roads_polylines)

        tmp_out_3 = torch.zeros(batch_size * num_roads, tmp_out.shape[-1], dtype=tmp_out.dtype, device=tmp_in2.device)
        tmp_out_3[valid_roads] = tmp_out

        roadgraph_polyline_features = tmp_out_3.view(batch_size, num_roads, tmp_out.shape[-1])
        ################################################################

        agent_feature_mask = agents_polylines_mask.any(2)
        roadgraph_feature_mask = roadgraph_polylines_mask.any(2)

        graph_in_features = torch.cat([
            agent_polyline_features, roadgraph_polyline_features
        ], dim=1)
        graph_mask = torch.cat([agent_feature_mask, roadgraph_feature_mask], dim=1)

        # print("graph_in_features", graph_in_features.shape)
        # print("graph_mask", graph_mask.shape)

        graph_out_features = graph_in_features.detach().clone()

        # graph_out_features = self.global_graph(graph_in_features, graph_mask)
        tmp_in = graph_in_features[graph_mask]

        offset = 0
        edge_indices_list = []
        for i in range(batch_size):
            num_nodes = graph_mask[i].sum().item()
            range_indices = torch.arange(num_nodes, dtype=torch.int64, device=graph_in_features.device)
            u, v = torch.meshgrid(range_indices, range_indices, indexing="ij")
            edge_indices = torch.stack([u, v], dim=0)
            edge_indices = edge_indices.flatten(1) + offset
            # print("edge_indices", edge_indices.shape)
            edge_indices_list.append(edge_indices)
            offset += num_nodes

        edge_indices = torch.cat(edge_indices_list, dim=-1)
        # print("edge_indices_list", edge_indices.shape)

        tmp_out = self.global_graph(tmp_in, edge_indices)

        graph_out_features[graph_mask] = tmp_out

        graph_features = torch.cat([graph_out_features, graph_in_features], dim=2)

        # tmp_inds = objects_of_interest.argmax(-1)
        # tmp = graph_features.gather(1, index=tmp_inds[:, None, None].repeat(1, 1, graph_features.shape[-1]))
        # tmp = tmp.squeeze(1)

        # pred_trajs, logits = self.decoder(tmp)
        # gt_trajs = targets.gather(1, index=tmp_inds[:, None, None, None].repeat(1, 1, targets.shape[-2], targets.shape[-1]))
        # gt_trajs = gt_trajs[..., :2]
        # gt_trajs = gt_trajs.squeeze(1)

        # gt_mask = targets_mask.gather(1, index=tmp_inds[:, None, None].repeat(1, 1, targets_mask.shape[-1]))
        # gt_mask = gt_mask.squeeze(1)

        # return pred_trajs, logits, gt_trajs, gt_mask

        trajs_list, probs_list, agent_inds_list = [], [], []
        for i in range(batch_size):
            agent_inds = torch.nonzero((objects_of_interest == 1)[i], as_tuple=True)[0]

            target_features = torch.cat([
                graph_features[i, agent_inds, :],
                agents_polylines[i, agent_inds, 0, :2],
                # agents_motion[i, agent_inds, :],
            ], dim=1)
            # target_features = graph_features[i, agent_inds, :]

            decoded_trajs, decoded_probs = self.decoder(target_features)
            decoded_probs = decoded_probs.sum(0)
            decoded_probs = decoded_probs - decoded_probs.max()
            decoded_probs = F.softmax(decoded_probs, dim=0)
            trajs_list.append(decoded_trajs)
            probs_list.append(decoded_probs)
            agent_inds_list.append(agent_inds)

        return trajs_list, probs_list, agent_inds_list

    def _training_and_validation_step(
        self,
        batch: Tuple[Polylines, Polylines, Polylines],
        batch_idx: int,
    ):
        (
            agents_polylines, agents_polylines_mask,
            roadgraph_polylines, roadgraph_polylines_mask,
            targets, targets_mask,
            tracks_to_predict, objects_of_interest,
            agents_motion,
        ) = batch

        # pred_trajs, logits, gt_trajs, gt_mask = self.forward(
        trajs_list, probs_list, agent_inds_list = self.forward(
            # agents_polylines=agents_polylines,
            # roadgraph_polylines=roadgraph_polylines,
            # edge_indices=edge_indices,
            # target_node_indices=target_node_indices,
            agents_polylines=agents_polylines,
            agents_polylines_mask=agents_polylines_mask,
            roadgraph_polylines=roadgraph_polylines,
            roadgraph_polylines_mask=roadgraph_polylines_mask,
            objects_of_interest=objects_of_interest,
            agents_motion=agents_motion,
            targets=targets, targets_mask=targets_mask,
        )

        # sq_diff = (pred_trajs - gt_trajs[:, None, :, :]) ** 2.
        # loss_ade, loss_ade_ce = compute_ade_losses(
        #     sq_diff, logits, gt_mask
        # )

        # loss = loss_ade + loss_ade_ce

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
        batch_size = batch[0].shape[0]
        loss, loss_ade, loss_ade_ce = self._training_and_validation_step(batch, batch)

        self.log(
            "train/loss", loss,
            on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size,
        )
        self.log(
            "train/loss_ade", loss_ade,
            on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size,
        )
        self.log(
            "train/loss_ade_ce", loss_ade_ce,
            on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size,
        )

        return loss

    def validation_step(self, batch, batch_idx: int):
        batch_size = batch[0].shape[0]
        loss, loss_ade, loss_ade_ce = self._training_and_validation_step(batch, batch)

        self.log(
            "val/loss", loss,
            on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size,
        )
        self.log(
            "val/loss_ade", loss_ade,
            on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size,
        )
        self.log(
            "val/loss_ade_ce", loss_ade_ce,
            on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size,
        )

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.learning_rate,
        )


# @torch.jit.script
def compute_ade_losses(
    sq_diff: torch.Tensor,
    logits: torch.Tensor,
    gt_mask: torch.Tensor,
):
    # sq_diff: N, M, T, 2
    # logits: N, M
    # gt_mask: N, T

    # N, M, T
    l2 = torch.sqrt(sq_diff.sum(-1))
    gt_mask = gt_mask[:, None, :].expand_as(l2).contiguous()
    # N, M
    ade = (l2 * gt_mask).sum(-1) / (gt_mask.sum(-1) + 1e-4)
    # ade = l2[gt_mask].mean(-1)
    # print("ade", ade.shape, ade.mean())
    # N
    min_ade, min_indices = ade.min(-1)
    # N
    ade_ce = -F.log_softmax(logits, dim=-1)
    min_ade_ce = ade_ce[min_indices]

    loss_ade = min_ade.mean()
    loss_ade_ce = min_ade_ce.mean()

    return loss_ade, loss_ade_ce


@torch.jit.script
def compute_neg_multi_log_likelihood(
    sq_diff: torch.Tensor,
    logits: torch.Tensor,
):
    # sq_diff: N, M, T, 2
    # logits: N, M

    # N, M, T
    error = sq_diff.sum(-1)
    # N, M
    error = F.log_softmax(logits, dim=-1) - 0.5 * error.sum(-1)
    # N
    error = -torch.logsumexp(error, dim=-1)

    return error.mean()
