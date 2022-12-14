from typing import Optional, List, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_scatter import scatter_max

from dl_playground.data.datasets.waymo_motion import Polylines
from dl_playground.layers.positional_encoding import (
    PositionalEncoding1D, PositionalEncoding2D
)


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

        self.agent_temporal_encoding = PositionalEncoding1D(32)
        self.target_motion_xy_encoding = PositionalEncoding2D(32)
        self.target_motion_yaw_encoding = PositionalEncoding1D(8)

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
        self.decoder = Decoder([2 * num_channels + 3 + 32 + 8, 4096, 512], num_modes=3, timesteps=80)

    def forward(
        self,
        agents_polylines: Polylines,
        roadgraph_polylines: Polylines,
        edge_indices: torch.Tensor,
        target_node_indices: torch.Tensor,
        target_current_states: torch.Tensor,
        agents_timestamp: torch.Tensor,
    ):
        agents_te = self.agent_temporal_encoding(agents_timestamp)

        agents_polylines.features = torch.cat([
            agents_polylines.features, agents_te
        ], dim=-1)

        agent_polyline_features = self.agent_subgraph(agents_polylines)
        roadgraph_polyline_features = self.roadmap_subgraph(roadgraph_polylines)

        graph_in_features = torch.cat([
            agent_polyline_features, roadgraph_polyline_features
        ], dim=0)
        graph_out_features = self.global_graph(graph_in_features, edge_indices)
        graph_features = torch.cat([graph_out_features, graph_in_features], dim=-1)

        target_current_motion_yaw = target_current_states[..., 5]

        target_current_motion_xy = self.target_motion_xy_encoding(
            target_current_states[..., 3:5]
        )
        target_current_motion_yaw = self.target_motion_yaw_encoding(
            target_current_states[..., 5]
        )

        target_agent_features = torch.cat([
            graph_features[target_node_indices],
            target_current_states[..., :3],
            target_current_motion_xy,
            target_current_motion_yaw,
        ], dim=-1)
        pred_trajs, logits = self.decoder(target_agent_features)

        return pred_trajs, logits

    def _training_and_validation_step(
        self,
        batch: Tuple[Polylines, Polylines, Polylines],
        batch_idx: int,
    ):
        (
            agents_polylines, roadgraph_polylines,
            target_current_states, target_future_states, target_future_mask,
            target_indices, target_node_indices, edge_indices,
            agents_timestamp,
        ) = batch

        pred_trajs, logits = self.forward(
            agents_polylines=agents_polylines,
            roadgraph_polylines=roadgraph_polylines,
            edge_indices=edge_indices,
            target_node_indices=target_node_indices,
            target_current_states=target_current_states,
            agents_timestamp=agents_timestamp,
        )

        gt_trajs = target_future_states[..., :2]

        sq_diff = (pred_trajs - gt_trajs[:, None, :, :]) ** 2.
        loss_ade, loss_ade_ce = compute_ade_losses(
            sq_diff, logits, target_future_mask
        )

        loss = loss_ade + loss_ade_ce

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
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.learning_rate,
        )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.max_epochs
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
        }


@torch.jit.script
def compute_ade_losses(
    sq_diff: torch.Tensor,
    logits: torch.Tensor,
    gt_mask: torch.Tensor,
):
    # sq_diff: B, M, T, 2
    # logits: B, M
    # gt_mask: B, T

    # B, M, T
    l2 = torch.sqrt(sq_diff.sum(-1))
    gt_mask = gt_mask[:, None, :].expand_as(l2)
    # B, M
    ade = (l2 * gt_mask).sum(-1) / (gt_mask.sum(-1) + 1e-4)
    # B
    min_ade, min_indices = ade.min(-1)
    # B
    ade_ce = -F.log_softmax(logits, dim=-1)
    min_ade_ce = ade_ce.gather(1, index=min_indices[:, None])

    loss_ade = min_ade.mean()
    loss_ade_ce = min_ade_ce.mean()

    return loss_ade, loss_ade_ce


@torch.jit.script
def compute_neg_multi_log_likelihood(
    sq_diff: torch.Tensor,
    logits: torch.Tensor,
):
    # sq_diff: B, M, T, 2
    # logits: B, M

    # N, M, T
    error = sq_diff.sum(-1)
    # N, M
    error = F.log_softmax(logits, dim=-1) - 0.5 * error.sum(-1)
    # N
    error = -torch.logsumexp(error, dim=-1)

    return error.mean()
