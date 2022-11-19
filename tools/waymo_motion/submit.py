import numpy as np
import torch
from tqdm import tqdm

from protos.motion_submission_pb2 import MotionChallengeSubmission


def main():
    outputs = torch.load("results/outputs.pt", map_location="cpu")

    timestamp_mask = np.arange(4, 81, 5)

    results = {}
    for pred_trajs, probs, meta_infos in tqdm(outputs):
        for pred_trajs_i, probs_i, meta_info in zip(pred_trajs, probs, meta_infos):
            scenario_id = meta_info["scenario_id"]
            target_agent_id = meta_info["target_agent_id"]
            target_current_xy = meta_info["target_current_xy"]
            rot_matrix = meta_info["rot_matrix"]

            pred_trajs_i = pred_trajs_i[:, timestamp_mask, :]

            pred_trajs_i = pred_trajs_i[:, :, None, :] @ np.linalg.inv(rot_matrix)[None, None, :, :]
            pred_trajs_i = pred_trajs_i[:, :, 0, :]
            pred_trajs_i = pred_trajs_i + target_current_xy[None, None, :]

            if scenario_id not in results:
                results[scenario_id] = []

            results[scenario_id].append((target_agent_id, pred_trajs_i, probs_i))

    submission = MotionChallengeSubmission()
    submission.account_name = "yang1013114323@gmail.com"
    submission.authors.extend(["vivym"])
    submission.submission_type = MotionChallengeSubmission.SubmissionType.MOTION_PREDICTION
    submission.unique_method_name = "dense_vectornet"

    for scenario_id, preds in tqdm(results.items()):
        scenario_predictions = submission.scenario_predictions.add()
        scenario_predictions.scenario_id = scenario_id
        prediction_set = scenario_predictions.single_predictions

        for target_agent_id, pred_trajs_i, probs_i in preds:
            predictions = prediction_set.predictions.add()
            predictions.object_id = target_agent_id

            for j in np.argsort(-probs_i):
                scored_trajectory = predictions.trajectories.add()
                scored_trajectory.confidence = probs_i[j]

                trajectory = scored_trajectory.trajectory
                pred_traj = pred_trajs_i[j]
                trajectory.center_x.extend(pred_traj[:, 0])
                trajectory.center_y.extend(pred_traj[:, 1])

    with open("results/submission.bin", "wb") as f:
        f.write(submission.SerializeToString())


if __name__ == "__main__":
    main()
