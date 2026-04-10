"""
Paradigm 3: multi-task learning (shared trunk, classification + regression heads).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from sensd_sers_analysis.config import (
    REGRESSION_MTL_BATCH_SIZE,
    REGRESSION_MTL_DROPOUT,
    REGRESSION_MTL_HIDDEN_DIMS,
    REGRESSION_MTL_LAMBDA_CLASSIFICATION,
    REGRESSION_MTL_LAMBDA_REGRESSION,
    REGRESSION_MTL_LEARNING_RATE,
    REGRESSION_MTL_MAX_EPOCHS,
    REGRESSION_MTL_PATIENCE,
    REGRESSION_MTL_VAL_FRACTION,
    REGRESSION_MTL_WEIGHT_DECAY,
    REGRESSION_RANDOM_STATE,
)
from sensd_sers_analysis.regression.metrics import regression_metrics

logger = logging.getLogger(__name__)


def serotype_class_labels_from_column(values: np.ndarray) -> tuple[str, ...]:
    """
    Sorted unique serotype strings for stable ``(0 .. N-1)`` class indices.

    Parameters
    ----------
    values:
        Serotype labels (e.g. ``df[class_col].to_numpy()``).

    Returns
    -------
    tuple[str, ...]
        Canonical label order used by :func:`encode_serotype_class_ids`.
    """
    uniq = pd.unique(pd.Series(values).astype(str))
    return tuple(sorted(s for s in uniq if s and str(s).lower() != "nan"))


def encode_serotype_class_ids(values: np.ndarray, class_labels: tuple[str, ...]) -> np.ndarray:
    """Map string serotype labels to integer class ids (same order as ``class_labels``)."""
    name_to_id = {name: i for i, name in enumerate(class_labels)}
    out = np.empty(len(values), dtype=np.int64)
    for i, raw in enumerate(values):
        key = str(raw).strip()
        if key not in name_to_id:
            raise ValueError(
                f"Unknown serotype label {key!r} for MTL; expected one of {class_labels}."
            )
        out[i] = name_to_id[key]
    return out


class MtlSpectralNet(nn.Module):
    """
    Shared MLP trunk with classification and regression heads.

    The regression head predicts ``log10`` concentration on the same scale as
    the sklearn baselines.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dims: tuple[int, ...],
        *,
        n_classes: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev, h),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev = h
        if layers:
            self.trunk: nn.Module = nn.Sequential(*layers)
            trunk_out = prev
        else:
            self.trunk = nn.Identity()
            trunk_out = in_dim
        self.head_cls = nn.Linear(trunk_out, n_classes)
        self.head_reg = nn.Linear(trunk_out, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        return self.head_cls(h), self.head_reg(h).squeeze(-1)


@dataclass
class MtlRegressionOutputs:
    """Held-out metrics and predictions for the MTL network."""

    y_true_reg: np.ndarray
    y_pred_reg: np.ndarray
    y_true_cls: np.ndarray
    y_pred_cls: np.ndarray
    rmse: float
    mae: float
    r2: float
    clf_accuracy: float
    model: MtlSpectralNet
    scaler: StandardScaler
    class_labels: tuple[str, ...]
    train_loss_history: Optional[list[float]] = None
    val_loss_history: Optional[list[float]] = None


def train_mtl_regressor(
    df: pd.DataFrame,
    feature_cols: list[str],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    target_col: str = "log_concentration",
    class_col: str = "target",
    random_state: int = REGRESSION_RANDOM_STATE,
) -> MtlRegressionOutputs:
    """
    Jointly train multi-class serotype classification and log-concentration regression.

    Uses a random sub-split of the **training** rows for early stopping. Test
    sensors never appear during training or validation.
    """
    available = [c for c in feature_cols if c in df.columns]
    if not available:
        raise ValueError(f"No feature columns found. Needed: {feature_cols}")

    X_all = df[available].fillna(0).to_numpy(dtype=np.float64, copy=False)
    y_reg_all = df[target_col].to_numpy(dtype=np.float64, copy=False)
    class_labels = serotype_class_labels_from_column(
        df[class_col].to_numpy(dtype=object, copy=False)
    )
    if len(class_labels) < 2:
        raise ValueError(
            "MTL classification head needs at least 2 serotype classes in the regression "
            f"subset; got {class_labels!r}."
        )
    n_classes = len(class_labels)
    y_cls_all = encode_serotype_class_ids(
        df[class_col].to_numpy(dtype=object, copy=False), class_labels
    )

    X_train_full = X_all[train_idx]
    y_reg_train_full = y_reg_all[train_idx]
    y_cls_train_full = y_cls_all[train_idx]

    X_test = X_all[test_idx]
    y_reg_test = y_reg_all[test_idx]
    y_cls_test = y_cls_all[test_idx]

    rng = np.random.RandomState(random_state + 17)
    n_tr = len(train_idx)
    n_val = max(1, int(n_tr * REGRESSION_MTL_VAL_FRACTION))
    perm = rng.permutation(n_tr)
    val_pos = perm[:n_val]
    tr_pos = perm[n_val:]
    if tr_pos.size == 0:
        tr_pos = val_pos
        val_pos = perm[: max(1, n_val // 2)]

    X_tr = X_train_full[tr_pos]
    y_reg_tr = y_reg_train_full[tr_pos]
    y_cls_tr = y_cls_train_full[tr_pos]
    X_val = X_train_full[val_pos]
    y_reg_val = y_reg_train_full[val_pos]
    y_cls_val = y_cls_train_full[val_pos]

    scaler_full = StandardScaler()
    X_train_scaled = scaler_full.fit_transform(X_train_full)
    X_tr_s = scaler_full.transform(X_tr)
    X_val_s = scaler_full.transform(X_val)
    X_test_scaled = scaler_full.transform(X_test)

    device = torch.device("cpu")
    in_dim = X_tr_s.shape[1]
    model = MtlSpectralNet(
        in_dim,
        tuple(REGRESSION_MTL_HIDDEN_DIMS),
        n_classes=n_classes,
        dropout=REGRESSION_MTL_DROPOUT,
    ).to(device)

    X_tr_t = torch.tensor(X_tr_s, dtype=torch.float32, device=device)
    y_reg_tr_t = torch.tensor(y_reg_tr, dtype=torch.float32, device=device)
    y_cls_tr_t = torch.tensor(y_cls_tr, dtype=torch.long, device=device)
    ds = TensorDataset(X_tr_t, y_cls_tr_t, y_reg_tr_t)
    batch_size = min(REGRESSION_MTL_BATCH_SIZE, len(ds))
    loader = DataLoader(ds, batch_size=max(1, batch_size), shuffle=True)

    X_val_t = torch.tensor(X_val_s, dtype=torch.float32, device=device)
    y_reg_val_t = torch.tensor(y_reg_val, dtype=torch.float32, device=device)
    y_cls_val_t = torch.tensor(y_cls_val, dtype=torch.long, device=device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=REGRESSION_MTL_LEARNING_RATE,
        weight_decay=REGRESSION_MTL_WEIGHT_DECAY,
    )
    loss_cls_fn = nn.CrossEntropyLoss()
    loss_reg_fn = nn.MSELoss()

    train_hist: list[float] = []
    val_hist: list[float] = []
    best_state: Optional[dict] = None
    best_val = float("inf")
    patience_left = REGRESSION_MTL_PATIENCE

    for epoch in range(REGRESSION_MTL_MAX_EPOCHS):
        model.train()
        epoch_losses: list[float] = []
        for xb, yc, yr in loader:
            opt.zero_grad(set_to_none=True)
            logits, yp = model(xb)
            loss = REGRESSION_MTL_LAMBDA_CLASSIFICATION * loss_cls_fn(logits, yc)
            loss = loss + REGRESSION_MTL_LAMBDA_REGRESSION * loss_reg_fn(yp, yr)
            loss.backward()
            opt.step()
            epoch_losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            lv, cv = model(X_val_t)
            v_loss = float(
                REGRESSION_MTL_LAMBDA_CLASSIFICATION * loss_cls_fn(lv, y_cls_val_t)
                + REGRESSION_MTL_LAMBDA_REGRESSION * loss_reg_fn(cv, y_reg_val_t)
            )
        train_hist.append(float(np.mean(epoch_losses)) if epoch_losses else 0.0)
        val_hist.append(v_loss)

        if v_loss < best_val - 1e-6:
            best_val = v_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = REGRESSION_MTL_PATIENCE
        else:
            patience_left -= 1
            if patience_left <= 0:
                logger.info("MTL early stopping at epoch %d", epoch + 1)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Refit on **all** training sensors with the early-stopped weights as init — one short
    # additional pass for stability on full train (optional). Per plan, evaluate on test only.
    full_ds = TensorDataset(
        torch.tensor(X_train_scaled, dtype=torch.float32, device=device),
        torch.tensor(y_cls_train_full, dtype=torch.long, device=device),
        torch.tensor(y_reg_train_full, dtype=torch.float32, device=device),
    )
    full_loader = DataLoader(
        full_ds,
        batch_size=min(REGRESSION_MTL_BATCH_SIZE, len(full_ds)),
        shuffle=True,
    )
    finetune_epochs = min(30, REGRESSION_MTL_MAX_EPOCHS // 4)
    for _ in range(max(1, finetune_epochs)):
        model.train()
        for xb, yc, yr in full_loader:
            opt.zero_grad(set_to_none=True)
            logits, yp = model(xb)
            loss = REGRESSION_MTL_LAMBDA_CLASSIFICATION * loss_cls_fn(logits, yc)
            loss = loss + REGRESSION_MTL_LAMBDA_REGRESSION * loss_reg_fn(yp, yr)
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        Xt = torch.tensor(X_test_scaled, dtype=torch.float32, device=device)
        logits, yp = model(Xt)
        pred_cls = torch.argmax(logits, dim=1).cpu().numpy()
        pred_reg = yp.cpu().numpy()

    rmse, mae, r2 = regression_metrics(y_reg_test, pred_reg)
    clf_acc = float(np.mean(pred_cls == y_cls_test))

    return MtlRegressionOutputs(
        y_true_reg=y_reg_test,
        y_pred_reg=pred_reg,
        y_true_cls=y_cls_test,
        y_pred_cls=pred_cls,
        rmse=rmse,
        mae=mae,
        r2=r2,
        clf_accuracy=clf_acc,
        model=model,
        scaler=scaler_full,
        class_labels=class_labels,
        train_loss_history=train_hist,
        val_loss_history=val_hist,
    )
