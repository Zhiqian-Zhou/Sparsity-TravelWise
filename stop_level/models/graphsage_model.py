"""
stop_level/models/graphsage_model.py
=============================================================================
GraphSAGE on the static station graph + per-stop MLP head.

Architecture:
  1. Two SAGEConv layers over the station adjacency graph compute
     `[N_stations, H]` embeddings from static station features.
  2. For each stop, look up the embedding of its station, concatenate with
     the tabular per-stop features, feed an MLP head → P(disrupted).

`torch_geometric` is imported lazily so the module loads even when PyG isn't
installed; calling `fit` then raises a clear ImportError.
"""
from __future__ import annotations
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stop_level.models.base import StopModel, n_neg_pos_ratio


# Static station features: features that are constant per-station, used as
# the input matrix to the GNN. The MLP head consumes the *remaining* features.
STATIC_STATION_FEATURES = (
    "lat", "lon", "degree", "betweenness_centrality", "avg_historical_delay",
)


def _require_pyg():
    try:
        import torch  # noqa
        import torch.nn as nn  # noqa
        import torch.nn.functional as Fnn  # noqa
        from torch_geometric.nn import SAGEConv  # noqa
        return torch, nn, Fnn, SAGEConv
    except ImportError as e:
        raise ImportError(
            "GraphSAGE requires torch + torch_geometric.\n"
            "    pip install torch torch_geometric"
        ) from e


class GraphSAGEStopModel(StopModel):
    name = "graphsage"

    def __init__(
        self,
        scenario: str,
        seed: int = 42,
        hidden: int = 64,
        head_hidden: int = 64,
        dropout: float = 0.2,
        epochs: int = 30,
        lr: float = 1e-3,
        batch_size: int = 8192,
        patience: int = 5,
        **_: Any,
    ) -> None:
        super().__init__(scenario, seed)
        self.hidden = hidden
        self.head_hidden = head_hidden
        self.dropout = dropout
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.patience = patience
        self.net = None  # torch.nn.Module after fit
        self.station_embed_input: np.ndarray | None = None
        self.tabular_cols: list[str] | None = None
        self.station_id_to_idx: dict | None = None

    def _build_station_input(self, ctx: dict) -> np.ndarray:
        """Static [N_stations, F_static] feature matrix for the GNN."""
        nodes_station = ctx["nodes_station"]
        sid_to_idx = ctx["station_id_to_idx"]
        # Order rows by idx
        rows = nodes_station.set_index("station_id").reindex(list(sid_to_idx.keys()))
        cols = [c for c in STATIC_STATION_FEATURES if c in rows.columns]
        if "betweenness_centrality" not in rows.columns and "betweenness" in ctx:
            rows = rows.assign(
                betweenness_centrality=rows.index.map(ctx["betweenness"])
            )
            cols.append("betweenness_centrality")
        X = rows[cols].fillna(0).astype("float32").values
        return X

    def fit(self, train_df, val_df, feat_cols, ctx=None, **_):
        torch, nn, Fnn, SAGEConv = _require_pyg()
        if ctx is None or "edge_index" not in ctx or "station_id_to_idx" not in ctx:
            raise ValueError("GraphSAGE needs ctx with edge_index, station_id_to_idx, nodes_station.")

        self.feat_cols = list(feat_cols)
        # Tabular features are everything in feat_cols that is NOT a station-static feature
        self.tabular_cols = [c for c in self.feat_cols if c not in STATIC_STATION_FEATURES]
        self.station_id_to_idx = ctx["station_id_to_idx"]
        self.station_embed_input = self._build_station_input(ctx)
        N_stations, F_static = self.station_embed_input.shape

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        edge_index = torch.tensor(ctx["edge_index"], dtype=torch.long, device=device)
        x_static   = torch.tensor(self.station_embed_input, dtype=torch.float, device=device)

        torch.manual_seed(self.seed)

        class Net(nn.Module):
            def __init__(self, F_static, F_tab, H, head_H, dropout):
                super().__init__()
                self.conv1 = SAGEConv(F_static, H, aggr="mean")
                self.conv2 = SAGEConv(H, H, aggr="mean")
                self.ln1   = nn.LayerNorm(H)
                self.ln2   = nn.LayerNorm(H)
                self.head  = nn.Sequential(
                    nn.Linear(H + F_tab, head_H),
                    nn.ReLU(), nn.Dropout(dropout),
                    nn.Linear(head_H, 1),
                )

            def station_embed(self, x, ei):
                h1 = Fnn.relu(self.ln1(self.conv1(x, ei)))
                h2 = Fnn.relu(self.ln2(self.conv2(h1, ei)))
                return h2

            def forward(self, x, ei, station_idx, tab):
                emb = self.station_embed(x, ei)
                e = emb[station_idx]
                z = torch.cat([e, tab], dim=-1)
                return self.head(z).squeeze(-1)

        net = Net(F_static, len(self.tabular_cols), self.hidden, self.head_hidden, self.dropout).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=1e-4)
        spw = n_neg_pos_ratio(train_df["y_stop"].values)
        bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([spw], dtype=torch.float, device=device))

        def make_tensors(df):
            sid_idx = df["station_id"].map(self.station_id_to_idx).fillna(-1).astype(int).values
            keep = sid_idx >= 0
            tab = df.loc[keep, self.tabular_cols].astype("float32").values
            y = df.loc[keep, "y_stop"].values.astype("float32")
            return (
                torch.tensor(sid_idx[keep], dtype=torch.long, device=device),
                torch.tensor(tab, dtype=torch.float, device=device),
                torch.tensor(y, dtype=torch.float, device=device),
                keep,
            )

        sid_tr, tab_tr, y_tr, _ = make_tensors(train_df)
        sid_va, tab_va, y_va, va_keep = make_tensors(val_df)

        best_pr_auc, best_state, no_improve = -1.0, None, 0
        n_train = sid_tr.shape[0]
        for epoch in range(1, self.epochs + 1):
            net.train()
            perm = torch.randperm(n_train, device=device)
            losses = []
            for i in range(0, n_train, self.batch_size):
                idx = perm[i:i+self.batch_size]
                opt.zero_grad()
                logits = net(x_static, edge_index, sid_tr[idx], tab_tr[idx])
                loss = bce(logits, y_tr[idx])
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                losses.append(loss.item())

            net.eval()
            with torch.no_grad():
                logits_va = net(x_static, edge_index, sid_va, tab_va)
                p_va = torch.sigmoid(logits_va).cpu().numpy()
            from sklearn.metrics import average_precision_score
            yv = y_va.cpu().numpy()
            pr_auc = float(average_precision_score(yv, p_va)) if yv.sum() else 0.0
            if pr_auc > best_pr_auc:
                best_pr_auc = pr_auc
                best_state  = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    break

        if best_state is not None:
            net.load_state_dict(best_state)
        self.net = net.cpu()

        # Tune threshold on val
        y_val_prob = self.predict_proba(val_df)
        self.tune_threshold(val_df["y_stop"].values, y_val_prob)
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        torch, *_ = _require_pyg()
        if self.net is None:
            raise RuntimeError("GraphSAGEStopModel.fit was not called.")
        device = next(self.net.parameters()).device
        x_static = torch.tensor(self.station_embed_input, dtype=torch.float, device=device)
        # We need edge_index for forward — store it on first call; here we
        # rebuild from a small in-memory state captured during fit.
        # For simplicity we re-encode rows where station is known:
        sid_idx = df["station_id"].map(self.station_id_to_idx).fillna(-1).astype(int).values
        tab = df[self.tabular_cols].astype("float32").values
        out = np.full(len(df), 0.5, dtype="float32")
        if not hasattr(self, "_edge_index_cache"):
            raise RuntimeError("GraphSAGE: edge_index cache missing — re-run fit before predict.")
        ei = self._edge_index_cache.to(device)
        self.net.eval()
        with torch.no_grad():
            keep = sid_idx >= 0
            sid_t = torch.tensor(sid_idx[keep], dtype=torch.long, device=device)
            tab_t = torch.tensor(tab[keep], dtype=torch.float, device=device)
            logits = self.net(x_static, ei, sid_t, tab_t)
            p = torch.sigmoid(logits).cpu().numpy()
        out[keep] = p.astype("float32")
        return out

    def fit_with_cache(self, train_df, val_df, feat_cols, ctx=None, **kwargs):
        """Fit + retain edge_index in memory for later predict_proba calls."""
        out = self.fit(train_df, val_df, feat_cols, ctx=ctx, **kwargs)
        torch, *_ = _require_pyg()
        self._edge_index_cache = torch.tensor(ctx["edge_index"], dtype=torch.long)
        return out

    def save(self, dir_path: Path) -> None:
        torch, *_ = _require_pyg()
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), dir_path / "graphsage.pt")
        with open(dir_path / "meta.pkl", "wb") as f:
            pickle.dump({
                "feat_cols": self.feat_cols,
                "tabular_cols": self.tabular_cols,
                "threshold": self.threshold_,
                "scenario": self.scenario,
                "station_id_to_idx": self.station_id_to_idx,
                "station_embed_input": self.station_embed_input,
                "hidden": self.hidden,
                "head_hidden": self.head_hidden,
                "dropout": self.dropout,
            }, f)

    @classmethod
    def load(cls, dir_path: Path, scenario: str) -> "GraphSAGEStopModel":
        # The fit method captures the architecture inline; reload requires a
        # ctx with edge_index. Callers should re-instantiate and call fit again,
        # OR persist ctx alongside the model. For simplicity we keep the
        # reload path minimal and document the limitation.
        raise NotImplementedError(
            "GraphSAGE reload requires re-supplying ctx (edge_index). Use "
            "the saved state_dict in graphsage.pt + meta.pkl manually."
        )
