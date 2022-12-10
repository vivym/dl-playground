import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from tqdm import tqdm

from dl_playground.data.datasets import WaymoMotionDataLoader
from dl_playground.models import DenseVectorNet


def main():
    dataloaders = WaymoMotionDataLoader(
        root_path="data/waymo_open_dataset_motion_v_1_1_0/preprocessed",
        use_rasterized_data=True,
        train_interval=1,
        val_interval=8,
        test_interval=1,
        train_batch_size=32,
        val_batch_size=1,
        test_batch_size=1,
        num_workers=16,
    )
    vis_dataloader = dataloaders.vis_dataloader()

    model = DenseVectorNet.load_from_checkpoint(
        "wandb/dl-playground/47da5cqu/checkpoints/epoch_059_loss_3.0299.ckpt"
    )
    model.eval()

    with torch.no_grad():
        for i, batch in tqdm(enumerate(vis_dataloader), total=1000):
            if i > 1000:
                break

            (
                agents_polylines, roadgraph_polylines,
                target_current_states, target_future_states, target_future_mask,
                target_indices, target_node_indices, edge_indices,
                agents_timestamp,
                rasterized_maps, gt_rasterized_trajs, gt_rasterized_masks,
                meta_infos,
                vectorized_maps,
            ) = batch
            meta_info = meta_infos[0]

            pred_trajs, logits = model(
                agents_polylines=agents_polylines,
                roadgraph_polylines=roadgraph_polylines,
                edge_indices=edge_indices,
                target_node_indices=target_node_indices,
                target_current_states=target_current_states,
                agents_timestamp=agents_timestamp,
                rasterized_maps=rasterized_maps,
            )
            pred_trajs = pred_trajs.squeeze(0)
            logits = logits.squeeze(0)

            probs = logits.softmax(dim=-1)
            sorted_indices = (-probs).argsort()

            probs = probs[sorted_indices]
            pred_trajs = pred_trajs[sorted_indices]

            V = vectorized_maps[0]

            X, idx = V[:, :44], V[:, 44].flatten()

            figure(figsize=(15, 15), dpi=300)
            for i in np.unique(idx):
                _X = X[idx == i]
                if _X[:, 5:12].sum() > 0:
                    plt.plot(_X[:, 0], _X[:, 1], linewidth=4, color="red")
                else:
                    plt.plot(_X[:, 0], _X[:, 1], color="black")
                plt.xlim([-224 // 4, 224 // 4])
                plt.ylim([-224 // 4, 224 // 4])

            gt_trajs = target_future_states[0, :, :2]
            gt_mask = target_future_mask[0]

            plt.plot(
                gt_trajs[gt_mask][::10, 0].numpy(),
                gt_trajs[gt_mask][::10, 1].numpy(),
                "-o",
                label="Ground Truth",
                markersize=12,
            )

            alphas = probs.clone()
            # alphas = alphas / alphas.max()
            alphas[0] = 0.8
            alphas[1:] /= alphas[1:].max()
            alphas[1:] *= 0.15
            for i in range(6):
                plt.plot(
                    pred_trajs[i][gt_mask][::10, 0].numpy(),
                    pred_trajs[i][gt_mask][::10, 1].numpy(),
                    "-o",
                    label=f"c{sorted_indices[i].item()} = {probs[i].item():.4f}",
                    alpha=alphas[i].item(),
                    markersize=12,
                )

            scenario_id = meta_info["scenario_id"]
            target_agent_id = meta_info["target_agent_id"]

            plt.title(f"{scenario_id}_{target_agent_id}")
            plt.legend(prop={"size": 18})
            plt.savefig(
                f"vis/{scenario_id}_{target_agent_id}.png",
                # dpi=300,
            )
            plt.close()


if __name__ == "__main__":
    main()
