from typing import Optional, List, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_scatter import scatter_max, scatter_mean

from dl_playground.data.datasets.waymo_motion import Polylines


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

        # def hook(module, grad_input, grad_output):
        #     # print("grad_input", grad_input[0].abs().mean())
        #     print("grad_output", grad_output[0].abs().mean())

        # self.fc.register_full_backward_hook(hook)

    def forward(self, x: Polylines):
        # print("x.features #1", x.features.abs().mean())
        x.features = self.fc(x.features)
        # print("x.features #2", x.features.abs().mean())
        # print("weights", self.fc.weight.abs().mean())
        # x.features = self.norm(x.features)
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
        self.decoder = Decoder([2 * num_channels + 6, 1024, 2048], num_modes=3, timesteps=80)

    def forward(
        self,
        agents_polylines: Polylines,
        roadgraph_polylines: Polylines,
        edge_indices: torch.Tensor,
        target_indices: torch.Tensor,
        target_node_indices: torch.Tensor,
        target_current_states: torch.Tensor,
    ):
        agent_polyline_features = self.agent_subgraph(agents_polylines)
        roadgraph_polyline_features = self.roadmap_subgraph(roadgraph_polylines)

        graph_in_features = torch.cat([
            agent_polyline_features, roadgraph_polyline_features
        ], dim=0)
        graph_out_features = self.global_graph(graph_in_features, edge_indices)
        graph_features = torch.cat([graph_out_features, graph_in_features], dim=-1)

        # print("graph_features", graph_features[target_node_indices].shape)
        # print("target_current_states", target_current_states.shape)

        target_agent_features = torch.cat([
            graph_features[target_node_indices],
            target_current_states[..., :6],
        ], dim=-1)
        pred_trajs, logits = self.decoder(target_agent_features)

        return pred_trajs, logits

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

    def _training_and_validation_step(
        self,
        batch: Tuple[Polylines, Polylines, Polylines],
        batch_idx: int,
    ):
        (
            agents_polylines, roadgraph_polylines,
            target_future_states, target_future_mask,
            target_indices, target_node_indices, edge_indices,
        ) = batch
        batch_size = agents_polylines.batch_size

        # torch.save(
        #     {
        #         "batch_size": agents_polylines.batch_size,
        #         "agents_polylines_features": agents_polylines.features,
        #         "agents_polylines_agg_indices": agents_polylines.agg_indices,
        #         "agents_polylines_batch_indices": agents_polylines.batch_indices,
        #         "roadgraph_polylines_features": roadgraph_polylines.features,
        #         "roadgraph_polylines_agg_indices": roadgraph_polylines.agg_indices,
        #         "roadgraph_polylines_batch_indices": roadgraph_polylines.batch_indices,
        #         "future_polylines_features": future_polylines.features,
        #         "future_polylines_agg_indices": future_polylines.agg_indices,
        #         "future_polylines_batch_indices": future_polylines.batch_indices,
        #         "future_agents_indices": future_agents_indices,
        #         "future_timestamp_mask": future_timestamp_mask,
        #         "edge_indices": edge_indices,
        #     },
        #     "batch.pth"
        # )
        # exit(0)

        target_current_states = target_future_states[..., 0, :]

        pred_trajs, logits = self.forward(
            agents_polylines=agents_polylines,
            roadgraph_polylines=roadgraph_polylines,
            edge_indices=edge_indices,
            target_indices=target_indices,
            target_node_indices=target_node_indices,
            target_current_states=target_current_states,
        )

        gt_trajs = target_future_states[..., 1:, :2]

        # loss_ade, loss_ade_ce = 0, 0
        # for i in range(batch_size):
        #     prob = logits[i:i + 1]
        #     prob = prob.sum(0)
        #     prob = prob - prob.max()
        #     prob = F.softmax(prob, dim=0)

        #     loss_ade_i, loss_ade_ce_i = self.loss_fn(pred_trajs[i:i + 1], prob, gt_trajs[i:i + 1])

        #     loss_ade += loss_ade_i
        #     loss_ade_ce += loss_ade_ce_i

        # loss_ade /= batch_size
        # loss_ade_ce /= batch_size

        sq_diff = (pred_trajs - gt_trajs[:, None, :, :]) ** 2.
        loss_ade, loss_ade_ce = compute_ade_losses(
            sq_diff, logits, target_future_mask
        )

        loss = loss_ade + loss_ade_ce

        # sq_diff = (
        #     (pred_trajs - gt_trajs[:, None, :, :]) * target_future_mask[:, None, :, None]
        # ) ** 2
        # sq_diff2 = (pred_trajs - gt_trajs[:, None, :, :])
        # sq_diff2 = sq_diff2 * target_future_mask[:, None, :, None]
        # loss = compute_neg_multi_log_likelihood(sq_diff, logits)

        # loss_ade, loss_ade_ce = 0., 0.

        return loss, loss_ade, loss_ade_ce

    def training_step(self, batch, batch_idx: int):
        batch_size = batch[0].batch_size
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
        batch_size = batch[0].batch_size
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
