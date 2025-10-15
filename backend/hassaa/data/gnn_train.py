import os
import json
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
from torch_geometric.loader import DataLoader

# Fix imports - use absolute imports for direct execution
try:
    # When running as module
    from .gnn_dataset import SetPieceGraphDataset
    from .models.gatv2_models import MultiTaskGATv2, SingleTaskGATv2Shot, SingleTaskGATv2Receiver
except ImportError:
    # When running directly
    from gnn_dataset import SetPieceGraphDataset
    from models.gatv2_models import MultiTaskGATv2, SingleTaskGATv2Shot, SingleTaskGATv2Receiver


def pairwise_ranking_loss(node_logits, y_receiver, batch, margin=0.5, negatives_per_pos=5):
    device = node_logits.device
    total_loss = torch.tensor(0.0, device=device)
    num_pairs = 0
    
    # Process each graph in the batch
    for g_id in batch.unique():
        mask = batch == g_id
        scores = node_logits[mask].squeeze(-1)
        labels = y_receiver[mask]
        
        # Find positive and negative indices
        pos_idx = (labels > 0.5).nonzero(as_tuple=False).flatten()
        neg_idx = (labels < 0.5).nonzero(as_tuple=False).flatten()
        
        # Skip if no positives or no negatives
        if pos_idx.numel() == 0 or neg_idx.numel() == 0:
            continue
            
        # Sample negatives for each positive (as specified in requirements)
        k = min(negatives_per_pos, neg_idx.numel())
        
        # For each positive, pair with sampled negatives
        for p in pos_idx:
            # Randomly sample k negatives
            perm = torch.randperm(neg_idx.numel(), device=device)[:k]
            sampled_negatives = neg_idx[perm]
            
            # Compute scores for positive and sampled negatives
            s_pos = scores[p].expand_as(sampled_negatives)
            s_neg = scores[sampled_negatives]
            
            # Apply margin ranking loss: max(0, margin - (s_pos - s_neg))
            loss = torch.relu(margin - (s_pos - s_neg)).mean()
            total_loss = total_loss + loss
            num_pairs += 1
    
    # Return average loss or zero if no pairs
    if num_pairs == 0:
        return torch.tensor(0.0, device=device)
    return total_loss / num_pairs


def precision_at_k(scores: np.ndarray, labels: np.ndarray, k: int) -> float:
    if len(scores) == 0:
        return 0.0
    idx = np.argsort(-scores)[:k]
    return float(labels[idx].sum() > 0)


def map_at_k(scores: np.ndarray, labels: np.ndarray, k: int) -> float:
    if len(scores) == 0 or labels.sum() == 0:
        return 0.0
    order = np.argsort(-scores)[:k]
    hits = 0.0
    ap = 0.0
    for i, idx in enumerate(order, start=1):
        if labels[idx] == 1:
            hits += 1
            ap += hits / i
    return ap / max(1.0, labels.sum()) if labels.sum() > 0 else 0.0


def ndcg_at_k(scores: np.ndarray, labels: np.ndarray, k: int) -> float:
    if len(scores) == 0:
        return 0.0
    order = np.argsort(-scores)[:k]
    dcg = 0.0
    for i, idx in enumerate(order, start=1):
        rel = 1.0 if labels[idx] == 1 else 0.0
        dcg += (2**rel - 1) / np.log2(i + 1)
    ideal = min(1, int(labels.sum()))
    idcg = 0.0
    for i in range(1, min(k, ideal) + 1):
        idcg += (2**1 - 1) / np.log2(i + 1)
    return dcg / idcg if idcg > 0 else 0.0


# Define constants for node feature indices
ATTACKER_FLAG_INDEX = 4  # is_attacker flag position in node features
GOALKEEPER_FLAG_INDEX = 5  # is_goalkeeper flag position

def get_attacker_mask(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Robustly extract attacker nodes with multiple fallback strategies + debug info."""
    attacker_mask = (x[:, ATTACKER_FLAG_INDEX] > 0.5) & mask
    debug_info = {"flag": torch.sum(attacker_mask).item()}

    # Fallback 1: non-goalkeeper nodes
    if not torch.any(attacker_mask):
        fallback_mask = (x[:, GOALKEEPER_FLAG_INDEX] <= 0.5) & mask
        if torch.any(fallback_mask):
            attacker_mask = fallback_mask
    debug_info["fallback_non_keeper"] = torch.sum(attacker_mask).item()

    # Fallback 2: all players except ball node
    if not torch.any(attacker_mask):
        graph_indices = torch.where(mask)[0]
        if len(graph_indices) > 1:
            attacker_mask = torch.zeros_like(mask, dtype=torch.bool)
            attacker_mask[graph_indices[:-1]] = True
    debug_info["fallback_all_except_ball"] = torch.sum(attacker_mask).item()

    # Log warning if no attackers found
    if not torch.any(attacker_mask):
        print(f"⚠️  No attackers found in graph with {torch.sum(mask).item()} nodes | Debug: {debug_info}")

    return attacker_mask

def evaluate_receiver_node_ranking(node_logits, batch, x, y_receiver) -> Dict[str, float]:
    """Evaluate receiver prediction with robust attacker detection and debugging"""
    metrics = {"top1": [], "top3": [], "top5": [], "map3": [], "ndcg3": []}
    
    # Debug counters
    total_graphs = 0
    graphs_with_attackers = 0
    total_attacker_nodes = 0
    
    for g_id in batch.unique():
        total_graphs += 1
        graph_mask = batch == g_id
        
        # Use robust attacker detection
        attacker_mask = get_attacker_mask(x, graph_mask)
        
        # Skip if still no valid nodes
        if not torch.any(attacker_mask):
            continue
            
        graphs_with_attackers += 1
        total_attacker_nodes += torch.sum(attacker_mask).item()
        
        # Get scores and labels for attackers
        scores = torch.sigmoid(node_logits[attacker_mask]).squeeze(-1).detach().cpu().numpy()
        labels = y_receiver[attacker_mask].detach().cpu().numpy()
        
        if len(scores) > 0:
            metrics["top1"].append(precision_at_k(scores, labels, 1))
            metrics["top3"].append(precision_at_k(scores, labels, 3))
            metrics["top5"].append(precision_at_k(scores, labels, 5))
            metrics["map3"].append(map_at_k(scores, labels, 3))
            metrics["ndcg3"].append(ndcg_at_k(scores, labels, 3))
    
    # Log debugging information every few calls
    if hasattr(evaluate_receiver_node_ranking, 'call_count'):
        evaluate_receiver_node_ranking.call_count += 1
    else:
        evaluate_receiver_node_ranking.call_count = 1
    
    if evaluate_receiver_node_ranking.call_count <= 3:  # Log first 3 calls
        print(f"🔍 RECEIVER EVAL DEBUG:")
        print(f"   Total graphs: {total_graphs}")
        print(f"   Graphs with attackers: {graphs_with_attackers}")
        print(f"   Total attacker nodes: {total_attacker_nodes}")
        print(f"   Feature shape: {x.shape}")
        print(f"   Metrics collected: {len(metrics['top3'])} values")
    
    return {k: float(np.mean(v)) if len(v) > 0 else 0.0 for k, v in metrics.items()}


def temperature_scale(logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 1000) -> float:
    # Single temperature for calibration, optimize NLL on validation set
    T = torch.nn.Parameter(torch.ones(1, device=logits.device))
    opt = torch.optim.LBFGS([T], lr=0.1, max_iter=50)

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(logits / T.clamp(min=1e-4), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(T.detach().item())


def compute_class_weights(y):
    """Compute positive weight for BCEWithLogitsLoss"""
    y_array = np.array(y)
    pos_count = y_array.sum()
    neg_count = len(y_array) - pos_count
    if pos_count == 0:
        return 1.0
    return float(neg_count / pos_count)


def filter_graphs_with_attackers(dataset, indices):
    """Filter dataset indices to only include graphs with attackers"""
    valid_indices = []
    for idx in indices:
        graph = dataset[idx]
        if graph.x.size(1) > ATTACKER_FLAG_INDEX:
            has_attackers = (graph.x[:, ATTACKER_FLAG_INDEX] > 0.5).any()
            if has_attackers:
                valid_indices.append(idx)
            else:
                # Fallback: check if has non-goalkeeper nodes
                if graph.x.size(1) > GOALKEEPER_FLAG_INDEX:
                    has_non_keepers = (graph.x[:, GOALKEEPER_FLAG_INDEX] <= 0.5).any()
                    if has_non_keepers:
                        valid_indices.append(idx)
    return valid_indices

def train_multitask_model(root: str, batch_size: int = 16, lr: float = 1e-3, hidden: int = 128, heads: int = 4, dropout: float = 0.3, radius: float = 15.0):
    """Train multi-task GATv2 model"""
    ds = SetPieceGraphDataset(root=root, is_train=True, radius_m=radius, save_sample_path=os.path.join(root, "sample_graph.json"))

    # GroupKFold by match_id to avoid leakage
    match_ids = [int(getattr(ds[i], "match_id", -1)) for i in range(len(ds))]
    indices = np.arange(len(ds))
    gkf = GroupKFold(n_splits=5)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    best_report = None
    fold_id = 0
    for train_idx, val_idx in gkf.split(indices, groups=match_ids):
        fold_id += 1
        
        # Filter indices to ensure graphs have attackers
        train_idx_filtered = filter_graphs_with_attackers(ds, train_idx)
        val_idx_filtered = filter_graphs_with_attackers(ds, val_idx)
        
        print(f"Fold {fold_id}: Original train={len(train_idx)}, val={len(val_idx)}")
        print(f"Fold {fold_id}: Filtered  train={len(train_idx_filtered)}, val={len(val_idx_filtered)}")
        
        if len(train_idx_filtered) == 0 or len(val_idx_filtered) == 0:
            print(f"Skipping fold {fold_id} - insufficient data after filtering")
            continue
            
        train_loader = DataLoader(ds[train_idx_filtered], batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(ds[val_idx_filtered], batch_size=batch_size)

        in_dim = ds[0].x.size(1)
        model = MultiTaskGATv2(in_dim=in_dim, hidden=hidden, heads=heads, dropout=dropout, edge_dim=4).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=8)

        # Compute class weights using filtered indices
        shot_labels = [ds[i].y_shot.item() for i in train_idx_filtered]
        receiver_labels = []
        for i in train_idx_filtered:
            receiver_labels.extend(ds[i].y_receiver.tolist())
        
        shot_pos_weight = compute_class_weights(shot_labels)
        receiver_pos_weight = compute_class_weights(receiver_labels)
        
        shot_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(shot_pos_weight, device=device))
        receiver_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(receiver_pos_weight, device=device))

        best_val_primary = -1
        patience = 25  # Increased patience to allow more training
        patience_counter = 0

        for epoch in range(1, 201):  # Full training with 200 epochs
            model.train()
            running_loss = 0.0
            recv_preds = []
            recv_labels = []
            shot_probs_epoch = []
            shot_labels_epoch = []
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                node_logits, shot_logits = model(batch)

                # Use improved pairwise ranking loss with margin=0.5 and 5 negatives per positive
                rank_loss = pairwise_ranking_loss(node_logits, batch.y_receiver, batch.batch, margin=0.5, negatives_per_pos=5)
                node_bce = receiver_criterion(node_logits.squeeze(), batch.y_receiver)
                # Weight the losses as specified (70% ranking, 30% BCE)
                receiver_loss = 0.7 * rank_loss + 0.3 * node_bce

                shot_loss = shot_criterion(shot_logits.squeeze(), batch.y_shot)
                loss = receiver_loss + shot_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                running_loss += float(loss.item())

                recv_preds.append(torch.sigmoid(node_logits).detach().cpu().numpy())
                recv_labels.append(batch.y_receiver.unsqueeze(1).cpu().numpy())
                shot_probs_epoch.append(torch.sigmoid(shot_logits).detach().cpu().numpy())
                shot_labels_epoch.append(batch.y_shot.view(-1, 1).cpu().numpy())

            # Flatten
            recv_preds = np.vstack(recv_preds)
            recv_labels = np.vstack(recv_labels)
            shot_probs_epoch = np.vstack(shot_probs_epoch)
            shot_labels_epoch = np.vstack(shot_labels_epoch)

            # Train metrics (coarse)
            train_recv_acc = accuracy_score(recv_labels, (recv_preds > 0.5).astype(int))
            train_recv_f1 = f1_score(recv_labels, (recv_preds > 0.5).astype(int), zero_division=0)
            train_shot_acc = accuracy_score(shot_labels_epoch, (shot_probs_epoch > 0.5).astype(int))
            train_shot_f1 = f1_score(shot_labels_epoch, (shot_probs_epoch > 0.5).astype(int), zero_division=0)

            # Validation
            model.eval()
            with torch.no_grad():
                all_node_logits = []
                all_y_receiver = []
                all_batch = []
                all_x = []
                shot_logits_list = []
                shot_labels_list = []
                for batch in val_loader:
                    batch = batch.to(device)
                    node_logits, shot_logits = model(batch)
                    all_node_logits.append(node_logits.cpu())
                    all_y_receiver.append(batch.y_receiver.cpu())
                    all_batch.append(batch.batch.cpu())
                    all_x.append(batch.x.cpu())
                    shot_logits_list.append(shot_logits.cpu())
                    shot_labels_list.append(batch.y_shot.view(-1, 1).cpu())

                node_logits_cat = torch.cat(all_node_logits, dim=0)
                y_receiver_cat = torch.cat(all_y_receiver, dim=0)
                batch_cat = torch.cat(all_batch, dim=0)
                x_cat = torch.cat(all_x, dim=0)
                shot_logits_cat = torch.cat(shot_logits_list, dim=0)
                shot_labels_cat = torch.cat(shot_labels_list, dim=0)

                # Receiver ranking metrics
                recv_rank = evaluate_receiver_node_ranking(node_logits_cat, batch_cat, x_cat, y_receiver_cat)

                # Shot metrics
                shot_probs = torch.sigmoid(shot_logits_cat).numpy()
                shot_labels_np = shot_labels_cat.numpy()
                try:
                    roc = roc_auc_score(shot_labels_np, shot_probs)
                except Exception:
                    roc = 0.5
                try:
                    pr = average_precision_score(shot_labels_np, shot_probs)
                except Exception:
                    pr = 0.0
                shot_acc = accuracy_score(shot_labels_np, (shot_probs > 0.5).astype(int))
                shot_f1 = f1_score(shot_labels_np, (shot_probs > 0.5).astype(int), zero_division=0)

                primary = 0.5 * recv_rank["top3"] + 0.5 * shot_f1
                # Don't rely solely on receiver metrics if they're all zero
                if recv_rank["top3"] == 0.0 and recv_rank["top1"] == 0.0:
                    primary = shot_f1  # Use shot F1 as primary metric
                scheduler.step(primary)
                
                # Print example predictions every 5 epochs
                if epoch % 5 == 0:
                    print(f"\n--- Example Predictions at Epoch {epoch} ---")
                    # Log attacker counts for debugging
                    attacker_count_log = []
                    example_count = 0
                    for batch_example in val_loader:
                        batch_example = batch_example.to(device)
                        node_logits_example, _ = model(batch_example)
                        for g_id in batch_example.batch.unique()[:3]:  # First 3 graphs
                            mask = batch_example.batch == g_id
                            attacker_mask = get_attacker_mask(batch_example.x, mask)
                            attacker_count = torch.sum(attacker_mask).item()
                            attacker_count_log.append(attacker_count)
                            
                            # Add extra debug logging after attacker_mask is determined
                            if torch.any(attacker_mask):
                                print(f"✅ Attacker nodes found: {torch.sum(attacker_mask).item()} for Graph {int(g_id)}")
                                scores = torch.sigmoid(node_logits_example[attacker_mask]).view(-1).cpu().numpy()
                                labels = batch_example.y_receiver[attacker_mask].cpu().numpy()
                                # Sort by scores (descending)
                                sorted_indices = np.argsort(-scores)
                                sorted_scores = scores[sorted_indices]
                                sorted_labels = labels[sorted_indices]
                                print(f"Graph {int(g_id)} - Attackers: {attacker_count}, Scores: {[f'{s:.3f}' for s in sorted_scores[:3]]}")
                                print(f"Graph {int(g_id)} - Labels: {sorted_labels[:3].tolist()}")
                            else:
                                print(f"❌ Still no attackers detected for Graph {int(g_id)}")
                                print(f"Graph {int(g_id)} - No attackers found (total nodes: {torch.sum(mask).item()})")
                            
                            example_count += 1
                            if example_count >= 3:
                                break
                        if example_count >= 3:
                            break
                    print(f"Attacker counts in validation: {attacker_count_log}")
                    print("--- End of Examples ---\n")

            if epoch % 5 == 0 or epoch == 50:
                print(f"Fold {fold_id} Epoch {epoch} | Loss {running_loss:.3f} | Recv Acc {train_recv_acc:.3f} F1 {train_recv_f1:.3f} | Shot Acc {train_shot_acc:.3f} F1 {train_shot_f1:.3f}")
                print(f"Val: Top1 {recv_rank['top1']:.3f} Top3 {recv_rank['top3']:.3f} MAP@3 {recv_rank['map3']:.3f} NDCG@3 {recv_rank['ndcg3']:.3f} | Shot F1 {shot_f1:.3f} ROC {roc:.3f} PR {pr:.3f}")

            if primary > best_val_primary:
                best_val_primary = primary
                patience_counter = 0
                # Save checkpoint
                ckpt_path = os.path.join(root, f"best_gnn_multitask_fold{fold_id}.pt")
                torch.save({
                    "model_state": model.state_dict(),
                    "in_dim": in_dim,
                    "hidden": hidden,
                    "heads": heads,
                    "dropout": dropout,
                }, ckpt_path)
                best_report = {
                    "fold": fold_id,
                    "receiver": recv_rank,
                    "shot": {"f1": float(shot_f1), "roc_auc": float(roc), "pr_auc": float(pr), "acc": float(shot_acc)},
                }
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping on fold {fold_id} at epoch {epoch}")
                    break

        # Temperature scaling for shot calibration on validation set
        model.eval()
        with torch.no_grad():
            all_logits = []
            all_labels = []
            for batch in val_loader:
                batch = batch.to(device)
                _, shot_logits = model(batch)
                all_logits.append(shot_logits)
                all_labels.append(batch.y_shot.view(-1, 1))
            logit_cat = torch.cat(all_logits, dim=0)
            label_cat = torch.cat(all_labels, dim=0)
        T = temperature_scale(logit_cat, label_cat)
        best_report["calibration_temperature_shot"] = T

        # Save per-event predictions JSON for the validation fold
        preds = []
        with torch.no_grad():
            batch_idx = 0
            for batch in val_loader:
                batch = batch.to(device)
                node_logits, shot_logits = model(batch)
                shot_prob = torch.sigmoid(shot_logits / T).view(-1).cpu().numpy()
                # Per-graph receiver ranking among attackers
                for g_id in batch.batch.unique():
                    mask = batch.batch == g_id
                    attacker_mask = get_attacker_mask(batch.x, mask)
                    
                    if torch.any(attacker_mask):
                        scores = torch.sigmoid(node_logits[attacker_mask]).view(-1).cpu().numpy()
                        node_indices = torch.nonzero(attacker_mask, as_tuple=False).view(-1).cpu().numpy().tolist()
                    else:
                        # Ultimate fallback: use empty scores if no attackers found
                        scores = np.array([])
                        node_indices = []
                    # Get corner_id from first node in this graph
                    first_node_idx = mask.nonzero(as_tuple=False)[0][0].item()
                    corner_id = -1
                    if hasattr(batch, "corner_id"):
                        # Check that corner_id exists and has the right shape
                        if batch.corner_id.numel() > first_node_idx:
                            corner_id_tensor = batch.corner_id[first_node_idx]
                            if corner_id_tensor.numel() == 1:
                                corner_id = int(corner_id_tensor.item())
                    
                    # Use batch_idx to access shot_prob since it corresponds to graph index in this batch
                    shot_confidence = float(shot_prob[batch_idx]) if batch_idx < len(shot_prob) else 0.0
                    preds.append({
                        "corner_id": corner_id,
                        "shot_confidence": shot_confidence,
                        "receiver_scores": {str(int(i)): float(s) for i, s in zip(node_indices, scores)},
                    })
                    batch_idx += 1

        with open(os.path.join(root, f"val_preds_multitask_fold{fold_id}.json"), "w", encoding="utf-8") as f:
            json.dump(preds, f)

        # Continue with all folds for complete cross-validation
        # break  # Removed to train all folds

    # Write final report
    if best_report is not None:
        with open(os.path.join(root, "gnn_multitask_report.json"), "w", encoding="utf-8") as f:
            json.dump(best_report, f, indent=2)
        print("Saved best multitask report:", best_report)
    else:
        print("No multitask report generated.")
    return best_report


def train_single_task_models(root: str, batch_size: int = 16, lr: float = 1e-3, hidden: int = 128, heads: int = 4, dropout: float = 0.3, radius: float = 15.0):
    """Train separate models for shot and receiver tasks"""
    ds = SetPieceGraphDataset(root=root, is_train=True, radius_m=radius, save_sample_path=os.path.join(root, "sample_graph.json"))

    # GroupKFold by match_id to avoid leakage
    match_ids = [int(getattr(ds[i], "match_id", -1)) for i in range(len(ds))]
    indices = np.arange(len(ds))
    gkf = GroupKFold(n_splits=5)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train shot model
    print("Training single-task shot model...")
    shot_report = None
    fold_id = 0
    for train_idx, val_idx in gkf.split(indices, groups=match_ids):
        fold_id += 1
        
        # Filter indices to ensure graphs have attackers
        train_idx_filtered = filter_graphs_with_attackers(ds, train_idx)
        val_idx_filtered = filter_graphs_with_attackers(ds, val_idx)
        
        print(f"Shot Model - Fold {fold_id}: Filtered train={len(train_idx_filtered)}, val={len(val_idx_filtered)}")
        
        if len(train_idx_filtered) == 0 or len(val_idx_filtered) == 0:
            print(f"Skipping shot fold {fold_id} - insufficient data after filtering")
            continue
            
        train_loader = DataLoader(ds[train_idx_filtered], batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(ds[val_idx_filtered], batch_size=batch_size)

        in_dim = ds[0].x.size(1)
        model = SingleTaskGATv2Shot(in_dim=in_dim, hidden=hidden, heads=heads, dropout=dropout, edge_dim=4).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=8)

        # Compute class weights using filtered indices
        shot_labels = [ds[i].y_shot.item() for i in train_idx_filtered]
        shot_pos_weight = compute_class_weights(shot_labels)
        shot_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(shot_pos_weight, device=device))

        best_val_f1 = -1
        patience = 25  # Increased patience to allow more training
        patience_counter = 0

        for epoch in range(1, 201):  # Full training with 200 epochs
            model.train()
            running_loss = 0.0
            shot_probs_epoch = []
            shot_labels_epoch = []
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                shot_logits = model(batch)
                shot_loss = shot_criterion(shot_logits.squeeze(), batch.y_shot)
                shot_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                running_loss += float(shot_loss.item())

                shot_probs_epoch.append(torch.sigmoid(shot_logits).detach().cpu().numpy())
                shot_labels_epoch.append(batch.y_shot.view(-1, 1).cpu().numpy())

            # Flatten
            shot_probs_epoch = np.vstack(shot_probs_epoch)
            shot_labels_epoch = np.vstack(shot_labels_epoch)

            # Train metrics
            train_shot_acc = accuracy_score(shot_labels_epoch, (shot_probs_epoch > 0.5).astype(int))
            train_shot_f1 = f1_score(shot_labels_epoch, (shot_probs_epoch > 0.5).astype(int), zero_division=0)

            # Validation
            model.eval()
            with torch.no_grad():
                shot_logits_list = []
                shot_labels_list = []
                for batch in val_loader:
                    batch = batch.to(device)
                    shot_logits = model(batch)
                    shot_logits_list.append(shot_logits.cpu())
                    shot_labels_list.append(batch.y_shot.view(-1, 1).cpu())

                shot_logits_cat = torch.cat(shot_logits_list, dim=0)
                shot_labels_cat = torch.cat(shot_labels_list, dim=0)

                # Shot metrics
                shot_probs = torch.sigmoid(shot_logits_cat).numpy()
                shot_labels_np = shot_labels_cat.numpy()
                try:
                    roc = roc_auc_score(shot_labels_np, shot_probs)
                except Exception:
                    roc = 0.5
                try:
                    pr = average_precision_score(shot_labels_np, shot_probs)
                except Exception:
                    pr = 0.0
                shot_acc = accuracy_score(shot_labels_np, (shot_probs > 0.5).astype(int))
                shot_f1 = f1_score(shot_labels_np, (shot_probs > 0.5).astype(int), zero_division=0)

                scheduler.step(shot_f1)

            if epoch % 5 == 0 or epoch == 50:
                print(f"Shot Model - Fold {fold_id} Epoch {epoch} | Loss {running_loss:.3f} | Shot Acc {train_shot_acc:.3f} F1 {train_shot_f1:.3f}")
                print(f"Shot Model - Val: Shot F1 {shot_f1:.3f} ROC {roc:.3f} PR {pr:.3f}")

            if shot_f1 > best_val_f1:
                best_val_f1 = shot_f1
                patience_counter = 0
                # Save checkpoint
                ckpt_path = os.path.join(root, f"best_gnn_shot_fold{fold_id}.pt")
                torch.save({
                    "model_state": model.state_dict(),
                    "in_dim": in_dim,
                    "hidden": hidden,
                    "heads": heads,
                    "dropout": dropout,
                }, ckpt_path)
                shot_report = {
                    "fold": fold_id,
                    "shot": {"f1": float(shot_f1), "roc_auc": float(roc), "pr_auc": float(pr), "acc": float(shot_acc)},
                }
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping on fold {fold_id} at epoch {epoch}")
                    break

        # Temperature scaling for shot calibration on validation set
        model.eval()
        with torch.no_grad():
            all_logits = []
            all_labels = []
            for batch in val_loader:
                batch = batch.to(device)
                shot_logits = model(batch)
                all_logits.append(shot_logits)
                all_labels.append(batch.y_shot.view(-1, 1))
            logit_cat = torch.cat(all_logits, dim=0)
            label_cat = torch.cat(all_labels, dim=0)
        T = temperature_scale(logit_cat, label_cat)
        shot_report["calibration_temperature_shot"] = T

        # Save per-event predictions JSON for the validation fold
        preds = []
        with torch.no_grad():
            batch_idx = 0
            for batch in val_loader:
                batch = batch.to(device)
                shot_logits = model(batch)
                shot_prob = torch.sigmoid(shot_logits / T).view(-1).cpu().numpy()
                # Per-graph predictions
                for g_id in batch.batch.unique():
                    mask = batch.batch == g_id
                    # Get corner_id from first node in this graph
                    first_node_idx = mask.nonzero(as_tuple=False)[0][0].item()
                    corner_id = -1
                    if hasattr(batch, "corner_id"):
                        # Check that corner_id exists and has the right shape
                        if batch.corner_id.numel() > first_node_idx:
                            corner_id_tensor = batch.corner_id[first_node_idx]
                            if corner_id_tensor.numel() == 1:
                                corner_id = int(corner_id_tensor.item())
                    
                    # Use batch_idx to access shot_prob since it corresponds to graph index in this batch
                    shot_confidence = float(shot_prob[batch_idx]) if batch_idx < len(shot_prob) else 0.0
                    preds.append({
                        "corner_id": corner_id,
                        "shot_confidence": shot_confidence,
                    })
                    batch_idx += 1

        with open(os.path.join(root, f"val_preds_shot_fold{fold_id}.json"), "w", encoding="utf-8") as f:
            json.dump(preds, f)

        # Continue with all folds for complete cross-validation
        # break  # Removed to train all folds

    # Train receiver model
    print("Training single-task receiver model...")
    receiver_report = None
    fold_id = 0
    for train_idx, val_idx in gkf.split(indices, groups=match_ids):
        fold_id += 1
        
        # Filter indices to ensure graphs have attackers
        train_idx_filtered = filter_graphs_with_attackers(ds, train_idx)
        val_idx_filtered = filter_graphs_with_attackers(ds, val_idx)
        
        print(f"Receiver Model - Fold {fold_id}: Filtered train={len(train_idx_filtered)}, val={len(val_idx_filtered)}")
        
        if len(train_idx_filtered) == 0 or len(val_idx_filtered) == 0:
            print(f"Skipping receiver fold {fold_id} - insufficient data after filtering")
            continue
            
        train_loader = DataLoader(ds[train_idx_filtered], batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(ds[val_idx_filtered], batch_size=batch_size)

        in_dim = ds[0].x.size(1)
        model = SingleTaskGATv2Receiver(in_dim=in_dim, hidden=hidden, heads=heads, dropout=dropout, edge_dim=4).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=8)

        # Compute class weights using filtered indices
        receiver_labels = []
        for i in train_idx_filtered:
            receiver_labels.extend(ds[i].y_receiver.tolist())
        receiver_pos_weight = compute_class_weights(receiver_labels)
        receiver_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(receiver_pos_weight, device=device))

        best_val_top3 = -1
        patience = 25  # Increased patience to allow more training
        patience_counter = 0

        for epoch in range(1, 201):  # Full training with 200 epochs
            model.train()
            running_loss = 0.0
            recv_preds = []
            recv_labels = []
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                node_logits = model(batch)

                # Use improved pairwise ranking loss with margin=0.5 and 5 negatives per positive
                rank_loss = pairwise_ranking_loss(node_logits, batch.y_receiver, batch.batch, margin=0.5, negatives_per_pos=5)
                # Also include BCE loss with positive weighting
                node_bce = receiver_criterion(node_logits.squeeze(), batch.y_receiver)
                # Weight the losses as specified (70% ranking, 30% BCE)
                receiver_loss = 0.7 * rank_loss + 0.3 * node_bce
                receiver_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                running_loss += float(receiver_loss.item())

                recv_preds.append(torch.sigmoid(node_logits).detach().cpu().numpy())
                recv_labels.append(batch.y_receiver.unsqueeze(1).cpu().numpy())

            # Flatten
            recv_preds = np.vstack(recv_preds)
            recv_labels = np.vstack(recv_labels)

            # Train metrics (coarse)
            train_recv_acc = accuracy_score(recv_labels, (recv_preds > 0.5).astype(int))
            train_recv_f1 = f1_score(recv_labels, (recv_preds > 0.5).astype(int), zero_division=0)

            # Validation
            model.eval()
            with torch.no_grad():
                all_node_logits = []
                all_y_receiver = []
                all_batch = []
                all_x = []
                for batch in val_loader:
                    batch = batch.to(device)
                    node_logits = model(batch)
                    all_node_logits.append(node_logits.cpu())
                    all_y_receiver.append(batch.y_receiver.cpu())
                    all_batch.append(batch.batch.cpu())
                    all_x.append(batch.x.cpu())

                node_logits_cat = torch.cat(all_node_logits, dim=0)
                y_receiver_cat = torch.cat(all_y_receiver, dim=0)
                batch_cat = torch.cat(all_batch, dim=0)
                x_cat = torch.cat(all_x, dim=0)

                # Receiver ranking metrics
                recv_rank = evaluate_receiver_node_ranking(node_logits_cat, batch_cat, x_cat, y_receiver_cat)
                
                # Print example predictions every 5 epochs
                if epoch % 5 == 0:
                    print(f"\n--- Receiver Example Predictions at Epoch {epoch} ---")
                    # Log attacker counts for debugging
                    attacker_count_log = []
                    example_count = 0
                    for batch_example in val_loader:
                        batch_example = batch_example.to(device)
                        node_logits_example = model(batch_example)
                        for g_id in batch_example.batch.unique()[:3]:  # First 3 graphs
                            mask = batch_example.batch == g_id
                            attacker_mask = get_attacker_mask(batch_example.x, mask)
                            attacker_count = torch.sum(attacker_mask).item()
                            attacker_count_log.append(attacker_count)
                            
                            # Add extra debug logging after attacker_mask is determined
                            if torch.any(attacker_mask):
                                print(f"✅ Attacker nodes found: {torch.sum(attacker_mask).item()} for Graph {int(g_id)}")
                                scores = torch.sigmoid(node_logits_example[attacker_mask]).view(-1).cpu().numpy()
                                labels = batch_example.y_receiver[attacker_mask].cpu().numpy()
                                # Sort by scores (descending)
                                sorted_indices = np.argsort(-scores)
                                sorted_scores = scores[sorted_indices]
                                sorted_labels = labels[sorted_indices]
                                print(f"Graph {int(g_id)} - Attackers: {attacker_count}, Scores: {[f'{s:.3f}' for s in sorted_scores[:3]]}")
                                print(f"Graph {int(g_id)} - Labels: {sorted_labels[:3].tolist()}")
                            else:
                                print(f"❌ Still no attackers detected for Graph {int(g_id)}")
                                print(f"Graph {int(g_id)} - No attackers found (total nodes: {torch.sum(mask).item()})")
                            
                            example_count += 1
                            if example_count >= 3:
                                break
                        if example_count >= 3:
                            break
                    print(f"Receiver - Attacker counts in validation: {attacker_count_log}")
                    print("--- End of Examples ---\n")

                primary = recv_rank["top3"]
                scheduler.step(primary)

            if epoch % 5 == 0 or epoch == 50:
                print(f"Receiver Model - Fold {fold_id} Epoch {epoch} | Loss {running_loss:.3f} | Recv Acc {train_recv_acc:.3f} F1 {train_recv_f1:.3f}")
                print(f"Receiver Model - Val: Top1 {recv_rank['top1']:.3f} Top3 {recv_rank['top3']:.3f} MAP@3 {recv_rank['map3']:.3f} NDCG@3 {recv_rank['ndcg3']:.3f}")

            if primary > best_val_top3:
                best_val_top3 = primary
                patience_counter = 0
                # Save checkpoint
                ckpt_path = os.path.join(root, f"best_gnn_receiver_fold{fold_id}.pt")
                torch.save({
                    "model_state": model.state_dict(),
                    "in_dim": in_dim,
                    "hidden": hidden,
                    "heads": heads,
                    "dropout": dropout,
                }, ckpt_path)
                receiver_report = {
                    "fold": fold_id,
                    "receiver": recv_rank,
                }
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping on fold {fold_id} at epoch {epoch}")
                    break

        # Save per-event predictions JSON for the validation fold
        preds = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                node_logits = model(batch)
                # Per-graph receiver ranking among attackers
                for g_id in batch.batch.unique():
                    mask = batch.batch == g_id
                    attacker_mask = (batch.x[:, 4] > 0.5) & mask
                    scores = torch.sigmoid(node_logits[attacker_mask]).view(-1).cpu().numpy()
                    node_indices = torch.nonzero(attacker_mask, as_tuple=False).view(-1).cpu().numpy().tolist()
                    # Get corner_id from first node in this graph
                    first_node_idx = mask.nonzero(as_tuple=False)[0][0].item()
                    corner_id = -1
                    if hasattr(batch, "corner_id"):
                        # Check that corner_id exists and has the right shape
                        if batch.corner_id.numel() > first_node_idx:
                            corner_id_tensor = batch.corner_id[first_node_idx]
                            if corner_id_tensor.numel() == 1:
                                corner_id = int(corner_id_tensor.item())
                    
                    preds.append({
                        "corner_id": corner_id,
                        "receiver_scores": {str(int(i)): float(s) for i, s in zip(node_indices, scores)},
                    })

        with open(os.path.join(root, f"val_preds_receiver_fold{fold_id}.json"), "w", encoding="utf-8") as f:
            json.dump(preds, f)

        # Continue with all folds for complete cross-validation
        # break  # Removed to train all folds

    # Write final reports
    if shot_report is not None:
        with open(os.path.join(root, "gnn_shot_report.json"), "w", encoding="utf-8") as f:
            json.dump(shot_report, f, indent=2)
        print("Saved shot report:", shot_report)
    
    if receiver_report is not None:
        with open(os.path.join(root, "gnn_receiver_report.json"), "w", encoding="utf-8") as f:
            json.dump(receiver_report, f, indent=2)
        print("Saved receiver report:", receiver_report)
    
    return shot_report, receiver_report


def compare_models_and_report(tabular_report_path: str, gnn_multitask_report=None, gnn_shot_report=None, gnn_receiver_report=None):
    """Compare GNN models with tabular baseline and generate final report"""
    # Load tabular baseline report
    try:
        with open(tabular_report_path, "r") as f:
            tabular_report = json.load(f)
    except Exception as e:
        print(f"Could not load tabular report: {e}")
        tabular_report = {}

    # Compare results
    comparison = {
        "tabular_baseline": tabular_report,
        "gnn_multitask": gnn_multitask_report or {},
        "gnn_single_task": {
            "shot": gnn_shot_report or {},
            "receiver": gnn_receiver_report or {}
        }
    }

    # Determine which model performs best
    best_model = "tabular"
    best_shot_f1 = tabular_report.get("shot", {}).get("f1", 0) if tabular_report else 0
    best_receiver_top3 = tabular_report.get("receiver", {}).get("top3", 0) if tabular_report else 0

    # Check multitask model
    if gnn_multitask_report:
        multitask_shot_f1 = gnn_multitask_report.get("shot", {}).get("f1", 0)
        multitask_receiver_top3 = gnn_multitask_report.get("receiver", {}).get("top3", 0)
        if multitask_shot_f1 > best_shot_f1 or multitask_receiver_top3 > best_receiver_top3:
            best_model = "gnn_multitask"
            best_shot_f1 = max(best_shot_f1, multitask_shot_f1)
            best_receiver_top3 = max(best_receiver_top3, multitask_receiver_top3)

    # Check single task models
    if gnn_shot_report:
        single_shot_f1 = gnn_shot_report.get("shot", {}).get("f1", 0)
        if single_shot_f1 > best_shot_f1:
            best_model = "gnn_single_task_shot"
            best_shot_f1 = single_shot_f1

    if gnn_receiver_report:
        single_receiver_top3 = gnn_receiver_report.get("receiver", {}).get("top3", 0)
        if single_receiver_top3 > best_receiver_top3:
            best_model = "gnn_single_task_receiver"
            best_receiver_top3 = single_receiver_top3

    comparison["best_model"] = best_model
    comparison["recommendation"] = (
        f"Based on evaluation metrics, the {best_model} model performs best. "
        f"Shot prediction F1: {best_shot_f1:.3f}, Receiver prediction Top-3: {best_receiver_top3:.3f}"
    )

    # Save comparison report
    with open(os.path.join(os.path.dirname(tabular_report_path), "model_comparison_report.json"), "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    print("\n=== MODEL COMPARISON REPORT ===")
    print(json.dumps(comparison, indent=2))
    return comparison


def train_and_eval(root: str, batch_size: int = 16, lr: float = 1e-3, hidden: int = 128, heads: int = 4, dropout: float = 0.2, radius: float = 15.0):
    """Main training function that trains all model variants and compares them"""
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Train multitask model
    print("=== TRAINING MULTITASK GATV2 MODEL ===")
    multitask_report = train_multitask_model(root, batch_size, lr, hidden, heads, dropout, radius)
    
    # Train single task models
    print("=== TRAINING SINGLE TASK GATV2 MODELS ===")
    shot_report, receiver_report = train_single_task_models(root, batch_size, lr, hidden, heads, dropout, radius)
    
    # Compare with tabular baseline
    print("=== COMPARING MODELS ===")
    tabular_report_path = os.path.join(root, "tabular_report.json")
    compare_models_and_report(tabular_report_path, multitask_report, shot_report, receiver_report)


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    here = os.path.dirname(__file__)
    train_and_eval(here)