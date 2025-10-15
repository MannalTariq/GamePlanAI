import os
import math
import json
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset


PITCH_X = 100.0
PITCH_Y = 100.0
GOAL_X = 100.0
GOAL_Y = 50.0


def _safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return default
        return float(value)
    except Exception:
        return default


def _z_score(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    if std == 0:
        return values - mean
    return (values - mean) / std


class SetPieceGraphDataset(InMemoryDataset):
    def __init__(
        self,
        root: str,
        split_indices: Optional[List[int]] = None,
        radius_m: float = 15.0,
        knn_k: int = 4,
        is_train: bool = True,
        normalize_continuous: bool = True,
        save_sample_path: Optional[str] = None,
    ) -> None:
        self.radius_m = radius_m
        self.knn_k = knn_k
        self.is_train = is_train
        self.normalize_continuous = normalize_continuous
        self.save_sample_path = save_sample_path

        super().__init__(root)

        processed_path = os.path.join(self.processed_dir, "graphs.pt")
        meta_path = os.path.join(self.processed_dir, "normalizers.npz")

        corner_path = os.path.join(self.root, "processed_csv", "corner_data.csv")
        freekick_path = os.path.join(self.root, "processed_csv", "freekick_data.csv")
        players_path = os.path.join(self.root, "processed_csv", "player_positions.csv")
        markings_path = os.path.join(self.root, "processed_csv", "marking_assignments.csv")

        if not (os.path.exists(corner_path) and os.path.exists(players_path) and os.path.exists(markings_path)):
            raise FileNotFoundError("Processed CSVs not found under data/processed_csv. Run preprocessing first.")

        corner_df = pd.read_csv(corner_path)
        players_df = pd.read_csv(players_path)
        markings_df = pd.read_csv(markings_path)
        # Freekicks optional
        freekick_df = None
        if os.path.exists(freekick_path):
            try:
                freekick_df = pd.read_csv(freekick_path)
            except Exception:
                freekick_df = None

        # Select events. Currently corners only for labels availability
        events = corner_df.copy()

        # Build indices list
        all_indices = list(range(len(events)))
        if split_indices is None:
            split_indices = all_indices

        data_list: List[Data] = []
        cont_stats: Dict[str, Tuple[float, float]] = {}

        # Pre-compute continuous fields to z-score across TRAIN only
        if self.normalize_continuous and self.is_train:
            cont_values = {
                "x": [],
                "y": [],
                "vx": [],
                "vy": [],
                "dist_ball": [],
                "dist_goal": [],
            }
        else:
            cont_values = None

        # Helper to build per-event graph
        def build_graph(ev_row: pd.Series) -> Optional[Data]:
            corner_id = ev_row.get("corner_id")
            ev_players = players_df[players_df["corner_id"] == corner_id]
            if len(ev_players) == 0:
                return None

            # Ball node (last index)
            ball_x = _safe_float(ev_row.get("contact_location_x"), _safe_float(ev_row.get("header_location_x"), _safe_float(ev_row.get("x"), 0.0)))
            ball_y = _safe_float(ev_row.get("contact_location_y"), _safe_float(ev_row.get("header_location_y"), _safe_float(ev_row.get("y"), 50.0)))

            # Normalize to [0,1]
            ball_xn = np.clip(ball_x / PITCH_X, 0.0, 1.0)
            ball_yn = np.clip(ball_y / PITCH_Y, 0.0, 1.0)

            # Player nodes
            node_feats: List[List[float]] = []
            team_vals: List[int] = []
            player_ids: List[int] = []
            positions: List[Tuple[float, float]] = []

            for _, prow in ev_players.iterrows():
                px = _safe_float(prow.get("x"))
                py = _safe_float(prow.get("y"))
                vx = _safe_float(prow.get("velocity_x"), default=None)
                vy = _safe_float(prow.get("velocity_y"), default=None)
                if vx is None or vy is None:
                    vm = _safe_float(prow.get("velocity_magnitude"))
                    vx, vy = vm, 0.0
                is_attacker = int(_safe_float(prow.get("is_attacking_team"), 0.0))
                is_goalkeeper = int(_safe_float(prow.get("is_goalkeeper"), 0.0))
                height_cm = _safe_float(prow.get("height_cm"))
                header_proxy = _safe_float(prow.get("header_ability_proxy"))
                team_encoding = _safe_float(prow.get("team_encoding"), is_attacker)
                role_encoding = _safe_float(prow.get("role_encoding"))
                pace = _safe_float(prow.get("pace"))
                jumping = _safe_float(prow.get("jumping"))
                heading = _safe_float(prow.get("heading"))
                minute_bucket = _safe_float(prow.get("minute_bucket"))

                # Normalize coordinates [0,1]
                xn = np.clip(px / PITCH_X, 0.0, 1.0)
                yn = np.clip(py / PITCH_Y, 0.0, 1.0)

                # Distances normalized by pitch diagonal
                dist_ball = math.hypot(px - ball_x, py - ball_y) / math.hypot(PITCH_X, PITCH_Y)
                dist_goal = math.hypot(PITCH_X - px, 50.0 - py) / math.hypot(PITCH_X, PITCH_Y)

                # Missing value handling - mark missing flags
                height_missing = 1.0 if np.isnan(height_cm) or height_cm == 0.0 else 0.0
                header_missing = 1.0 if np.isnan(header_proxy) or header_proxy == 0.0 else 0.0
                pace_missing = 1.0 if np.isnan(pace) or pace == 0.0 else 0.0
                jumping_missing = 1.0 if np.isnan(jumping) or jumping == 0.0 else 0.0
                heading_missing = 1.0 if np.isnan(heading) or heading == 0.0 else 0.0

                feat = [
                    xn, yn, vx, vy,
                    float(is_attacker), float(is_goalkeeper),
                    dist_ball, dist_goal,
                    role_encoding, height_cm, header_proxy,
                    team_encoding, pace, jumping, heading,
                    minute_bucket,
                    height_missing, header_missing, pace_missing, jumping_missing, heading_missing
                ]
                node_feats.append(feat)
                team_vals.append(is_attacker)
                player_ids.append(int(_safe_float(prow.get("player_id"), -1)))
                positions.append((px, py))

            # Ball node features appended to end
            delivery_type = str(ev_row.get("delivery_type")) if "delivery_type" in ev_row else "unknown"
            delivery_target = _safe_float(ev_row.get("delivery_target"))
            # crude speed proxy as mean of |vx| over players
            avg_speed = float(np.mean([abs(f[2]) + abs(f[3]) for f in node_feats]) / 2.0) if node_feats else 0.0
            
            # Delivery type encoding (inswing/outswing)
            delivery_type_encoding = 1.0 if delivery_type.lower() == "inswing" else 0.0
            
            ball_feat = [
                ball_xn, ball_yn, avg_speed, delivery_type_encoding,
                0.0, 0.0,  # is_attacker, is_goalkeeper
                0.0, 0.0,  # dist_ball, dist_goal (not meaningful for ball)
                0.0, 0.0, 0.0,  # role, height, header
                0.0, 0.0, 0.0, 0.0,  # team, pace, jumping, heading
                0.0,  # minute_bucket
                0.0, 0.0, 0.0, 0.0, 0.0  # missing flags
            ]
            node_feats.append(ball_feat)

            # Edges: within radius and to ball
            num_players = len(player_ids)
            ball_idx = num_players  # last
            edge_index: List[Tuple[int, int]] = []
            edge_attr: List[List[float]] = []

            # Connect to ball
            for i in range(num_players):
                px, py = positions[i]
                dist = math.hypot(px - ball_x, py - ball_y)
                angle_goal = math.atan2(GOAL_Y - py, GOAL_X - px)
                same_team = 0.0  # ball
                marking_flag = 0.0
                # Add distance to ball and angle to goal as additional edge features
                dist_to_ball = dist / math.hypot(PITCH_X, PITCH_Y)  # Normalized distance
                angle_to_goal = angle_goal  # Angle to goal from player position
                edge_index.append((i, ball_idx))
                edge_index.append((ball_idx, i))
                edge_attr.append([dist, angle_goal, same_team, marking_flag, dist_to_ball, angle_to_goal])
                edge_attr.append([dist, angle_goal, same_team, marking_flag, dist_to_ball, angle_to_goal])

            # Player-player edges by radius
            R = self.radius_m
            for i in range(num_players):
                xi, yi = positions[i]
                for j in range(i + 1, num_players):
                    xj, yj = positions[j]
                    d = math.hypot(xi - xj, yi - yj)
                    if d <= R:
                        ang_i = math.atan2(GOAL_Y - yi, GOAL_X - xi)
                        ang_j = math.atan2(GOAL_Y - yj, GOAL_X - xj)
                        same_team = 1.0 if team_vals[i] == team_vals[j] else 0.0
                        # marking flag if exists
                        marking_flag = 0.0
                        # Check if there's a marking assignment between these players
                        if not markings_df.empty:
                            # Look for marking assignments for this corner_id and these player_ids
                            player_markings = markings_df[
                                (markings_df["corner_id"] == corner_id) & 
                                ((markings_df["defender_id"] == player_ids[i]) | (markings_df["defender_id"] == player_ids[j])) & 
                                ((markings_df["attacker_id"] == player_ids[i]) | (markings_df["attacker_id"] == player_ids[j]))
                            ]
                            if not player_markings.empty:
                                marking_flag = 1.0
                        # Add distance to ball and angle to goal for player-player edges (set to 0 for these)
                        edge_index.append((i, j))
                        edge_index.append((j, i))
                        edge_attr.append([d, ang_i, same_team, marking_flag, 0.0, ang_i])
                        edge_attr.append([d, ang_j, same_team, marking_flag, 0.0, ang_j])

            # kNN to ensure connectivity (based on Euclidean distance)
            if self.knn_k and self.knn_k > 0 and num_players > 1:
                coords = np.array(positions)
                for i in range(num_players):
                    dists = np.linalg.norm(coords - coords[i], axis=1)
                    order = np.argsort(dists)
                    k_targets = [j for j in order[1:self.knn_k + 1]]
                    for j in k_targets:
                        d = float(dists[j])
                        ang_i = math.atan2(GOAL_Y - positions[i][1], GOAL_X - positions[i][0])
                        ang_j = math.atan2(GOAL_Y - positions[j][1], GOAL_X - positions[j][0])
                        same_team = 1.0 if team_vals[i] == team_vals[j] else 0.0
                        edge_index.append((i, j))
                        edge_index.append((j, i))
                        # Add distance to ball and angle to goal for kNN edges (set to 0 for these)
                        edge_attr.append([d, ang_i, same_team, 0.0, 0.0, ang_i])
                        edge_attr.append([d, ang_j, same_team, 0.0, 0.0, ang_j])

            x = torch.tensor(node_feats, dtype=torch.float)
            if len(edge_index) > 0:
                edge_index_t = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
                edge_attr_t = torch.tensor(edge_attr, dtype=torch.float)
            else:
                edge_index_t = torch.zeros((2, 0), dtype=torch.long)
                # Update to 6 features
                edge_attr_t = torch.zeros((0, 6), dtype=torch.float)

            # Labels
            # Receiver node: closest to header location if provided
            receiver_labels = torch.zeros(x.size(0), dtype=torch.float)
            if np.isfinite(_safe_float(ev_row.get("header_location_x"))) and np.isfinite(_safe_float(ev_row.get("header_location_y"))):
                hx = _safe_float(ev_row.get("header_location_x"))
                hy = _safe_float(ev_row.get("header_location_y"))
                best_idx = None
                best_d = 1e9
                for i in range(num_players):
                    d = math.hypot(positions[i][0] - hx, positions[i][1] - hy)
                    if d < best_d:
                        best_d = d
                        best_idx = i
                if best_idx is not None:
                    receiver_labels[best_idx] = 1.0
            # 3️⃣ Sanity Debug Print - Added debug print
            print(f"corner_id={corner_id}, pos_index={torch.where(receiver_labels == 1)[0].tolist()}, total_nodes={len(node_feats)}")

            is_shot = _safe_float(ev_row.get("is_shot"), _safe_float(ev_row.get("is_goal"), 0.0))
            y_graph = torch.tensor([is_shot], dtype=torch.float)

            data = Data(
                x=x,
                edge_index=edge_index_t,
                edge_attr=edge_attr_t,
                y_shot=y_graph,
                y_receiver=receiver_labels,
                num_nodes=x.size(0),
            )
            # Attach some meta
            data.corner_id = int(_safe_float(corner_id, -1))
            data.match_id = int(_safe_float(ev_row.get("match_id"), -1))
            data.ball_index = int(ball_idx)
            return data

        # Build graphs
        for idx in split_indices:
            g = build_graph(events.iloc[idx])
            if g is not None:
                data_list.append(g)

        # Save sample graph JSON
        if self.save_sample_path and len(data_list) > 0:
            g0 = data_list[0]
            sample = {
                "num_nodes": int(g0.num_nodes),
                "num_edges": int(g0.edge_index.size(1)),
                "ball_index": int(g0.ball_index),
                "x_head": g0.x[: min(5, g0.x.size(0))].tolist(),
                "edge_attr_head": g0.edge_attr[: min(5, g0.edge_attr.size(0))].tolist(),
                "y_shot": float(g0.y_shot.item()),
                "pos_receiver": int(torch.argmax(g0.y_receiver).item()) if g0.y_receiver.sum() > 0 else -1,
            }
            with open(self.save_sample_path, "w", encoding="utf-8") as f:
                json.dump(sample, f, indent=2)

        self.data, self.slices = self.collate(data_list)