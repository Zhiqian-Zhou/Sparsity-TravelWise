"""
stop_level/models/bilstm_model.py
=============================================================================
BiLSTM stop-sequence model — the natural fit for stop-level disruption,
because cascading delays propagate forward through the route.

Architecture:
  Per-service input shape `[max_stops, F_per_stop]`.
  - Scenario A: bidirectional LSTM (1 layer, hidden 128) → per-stop head.
  - Scenario B: causal (unidirectional) LSTM with `prev_stop_actual_delay`
    injected as a feature at each timestep so the model is forced to
    respect the inflight constraint.

Variable-length services are right-padded and mask-aware.
"""
from __future__ import annotations
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stop_level.models.base import StopModel, n_neg_pos_ratio


def _require_torch():
    try:
        import torch  # noqa
        import torch.nn as nn  # noqa
        return torch, nn
    except ImportError as e:
        raise ImportError(
            "BiLSTM requires torch.\n    pip install torch"
        ) from e


class BiLSTMStopModel(StopModel):
    name = "bilstm"

    def __init__(
        self,
        scenario: str,
        seed: int = 42,
        hidden: int = 128,
        dropout: float = 0.2,
        epochs: int = 25,
        lr: float = 1e-3,
        batch_size: int = 256,        # services per batch
        patience: int = 4,
        max_stops: int = 80,
        **_: Any,
    ) -> None:
        super().__init__(scenario, seed)
        self.hidden = hidden
        self.dropout = dropout
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.patience = patience
        self.max_stops = max_stops
        self.net = None

    # ── Sequence packing helpers ────────────────────────────────────────────
    def _pack_sequences(self, df: pd.DataFrame):
        """
        Group rows by service_id, sort by stop_order, pad to max_stops.

        Returns
        -------
        x_pad     : [n_services, max_stops, F]   float32
        y_pad     : [n_services, max_stops]       float32
        mask      : [n_services, max_stops]       bool
        row_index : list of (service_id, original_row_indices)  for unpacking
        """
        df = df.sort_values(["service_id", "stop_order"]).reset_index()
        F = len(self.feat_cols)
        groups = list(df.groupby("service_id", sort=False))
        n = len(groups)
        L = self.max_stops
        x_pad = np.zeros((n, L, F), dtype="float32")
        y_pad = np.zeros((n, L), dtype="float32")
        mask  = np.zeros((n, L), dtype=bool)
        row_idx: list[tuple[str, np.ndarray]] = []
        for k, (sid, g) in enumerate(groups):
            T = min(len(g), L)
            x_pad[k, :T] = g[self.feat_cols].iloc[:T].astype("float32").values
            y_pad[k, :T] = g["y_stop"].iloc[:T].astype("float32").values
            mask[k, :T]  = True
            row_idx.append((sid, g["index"].iloc[:T].values))
        return x_pad, y_pad, mask, row_idx

    def fit(self, train_df, val_df, feat_cols, ctx=None, **_):
        torch, nn = _require_torch()
        self.feat_cols = list(feat_cols)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(self.seed)

        x_tr, y_tr, m_tr, _ = self._pack_sequences(train_df)
        x_va, y_va, m_va, _ = self._pack_sequences(val_df)
        F = x_tr.shape[-1]

        bidirectional = (self.scenario == "A")

        class SeqNet(nn.Module):
            def __init__(self, F, H, dropout, bidir):
                super().__init__()
                self.lstm = nn.LSTM(F, H, num_layers=1, batch_first=True,
                                     bidirectional=bidir, dropout=0.0)
                out_dim = H * (2 if bidir else 1)
                self.head = nn.Sequential(
                    nn.LayerNorm(out_dim),
                    nn.Linear(out_dim, out_dim), nn.ReLU(), nn.Dropout(dropout),
                    nn.Linear(out_dim, 1),
                )

            def forward(self, x):
                h, _ = self.lstm(x)
                return self.head(h).squeeze(-1)

        net = SeqNet(F, self.hidden, self.dropout, bidirectional).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=1e-4)
        spw = n_neg_pos_ratio(y_tr[m_tr])
        bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([spw], dtype=torch.float, device=device),
                                     reduction="none")

        x_tr_t = torch.from_numpy(x_tr).float().to(device)
        y_tr_t = torch.from_numpy(y_tr).float().to(device)
        m_tr_t = torch.from_numpy(m_tr).to(device)        # bool preserved
        x_va_t = torch.from_numpy(x_va).float().to(device)
        y_va_t = torch.from_numpy(y_va).float().to(device)
        m_va_t = torch.from_numpy(m_va).to(device)

        best_pr, best_state, stale = -1.0, None, 0
        n = x_tr_t.shape[0]
        from sklearn.metrics import average_precision_score
        for epoch in range(1, self.epochs + 1):
            net.train()
            perm = torch.randperm(n, device=device)
            for i in range(0, n, self.batch_size):
                idx = perm[i:i+self.batch_size]
                opt.zero_grad()
                logits = net(x_tr_t[idx])                  # [B, L]
                loss = bce(logits, y_tr_t[idx])
                loss = (loss * m_tr_t[idx]).sum() / m_tr_t[idx].sum().clamp(min=1)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()

            net.eval()
            with torch.no_grad():
                logits_va = net(x_va_t)
                p_va = torch.sigmoid(logits_va).cpu().numpy()
            yv_flat = y_va[m_va]
            pv_flat = p_va[m_va]
            pr_auc = float(average_precision_score(yv_flat, pv_flat)) if yv_flat.sum() else 0.0
            if pr_auc > best_pr:
                best_pr = pr_auc
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= self.patience:
                    break

        if best_state is not None:
            net.load_state_dict(best_state)
        self.net = net

        # Threshold tuning on val (flat over all stops)
        with torch.no_grad():
            logits_va = self.net(x_va_t)
            p_va = torch.sigmoid(logits_va).cpu().numpy()
        self.tune_threshold(y_va[m_va], p_va[m_va])
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        torch, _ = _require_torch()
        if self.net is None:
            raise RuntimeError("BiLSTMStopModel.fit was not called.")
        device = next(self.net.parameters()).device
        x_pad, _, mask, row_idx = self._pack_sequences(df)
        x_t = torch.tensor(x_pad, dtype=torch.float, device=device)
        self.net.eval()
        with torch.no_grad():
            logits = self.net(x_t)
            p = torch.sigmoid(logits).cpu().numpy()
        # Unpack back to df row order
        out = np.full(len(df), 0.5, dtype="float32")
        for k, (sid, original_idx) in enumerate(row_idx):
            T = mask[k].sum()
            out[original_idx] = p[k, :T].astype("float32")
        return out

    def save(self, dir_path: Path) -> None:
        torch, _ = _require_torch()
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), dir_path / "bilstm.pt")
        with open(dir_path / "meta.pkl", "wb") as f:
            pickle.dump({
                "feat_cols":  self.feat_cols,
                "threshold":  self.threshold_,
                "scenario":   self.scenario,
                "hidden":     self.hidden,
                "dropout":    self.dropout,
                "max_stops":  self.max_stops,
                "F":          len(self.feat_cols),
                "bidirectional": self.scenario == "A",
            }, f)

    @classmethod
    def load(cls, dir_path: Path, scenario: str) -> "BiLSTMStopModel":
        raise NotImplementedError(
            "BiLSTM reload requires reconstructing the SeqNet inner class. "
            "For evaluation, prefer running training and prediction in the "
            "same process."
        )
