"""
models/ml_models.py
XGBoost classifier and PyTorch MLP for W/D/L prediction.
Both expose: fit(X, y), predict_proba(X) → (p_win, p_draw, p_loss)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
import xgboost as xgb
import joblib
from pathlib import Path

from config import XGBOOST_CONFIG, NEURAL_NET_CONFIG, MODELS_DIR, ALL_FEATURES


# ── Label encoding: W=0, D=1, L=2 ─────────────────────────────────────────────
RESULT_TO_INT = {"W": 0, "D": 1, "L": 2}
INT_TO_RESULT = {0: "W", 1: "D", 2: "L"}


def encode_result(home_score, away_score) -> int:
    if home_score > away_score:
        return 0  # W
    elif home_score == away_score:
        return 1  # D
    else:
        return 2  # L


# ── XGBoost ────────────────────────────────────────────────────────────────────
class XGBoostMatchModel:
    """
    XGBoost multi-class classifier predicting P(W/D/L).
    Uses early stopping on validation log-loss.
    Cross-validated during training to select best iteration.
    """

    def __init__(self, config: dict = None, feature_names: list[str] = None):
        self.config = {**XGBOOST_CONFIG, **(config or {})}
        self.feature_names = feature_names or ALL_FEATURES
        self.model = None
        self.scaler = StandardScaler()
        self._best_ntree = None

    def _prep_X(self, df: pd.DataFrame) -> np.ndarray:
        cols = [f for f in self.feature_names if f in df.columns]
        X = df[cols].fillna(0.0).values.astype(float)
        return X

    def fit(self, df: pd.DataFrame, cv_folds: int = 5) -> "XGBoostMatchModel":
        """
        df must contain all feature columns + home_score + away_score.
        Performs stratified k-fold CV to find best number of trees,
        then retrains on full data.
        """
        X = self._prep_X(df)
        y = np.array([encode_result(r["home_score"], r["away_score"])
                       for _, r in df.iterrows()])

        X_scaled = self.scaler.fit_transform(X)

        # Cross-validation to find optimal n_estimators
        cv_scores = []
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True,
                               random_state=self.config["random_state"])

        print(f"XGBoost: {cv_folds}-fold CV on {len(X)} matches...")
        for fold, (tr_idx, val_idx) in enumerate(skf.split(X_scaled, y)):
            X_tr, X_val = X_scaled[tr_idx], X_scaled[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]

            cfg = {k: v for k, v in self.config.items()
                   if k not in ("early_stopping_rounds",)}
            model = xgb.XGBClassifier(**cfg, use_label_encoder=False)
            model.fit(X_tr, y_tr,
                      eval_set=[(X_val, y_val)],
                      verbose=False,)

            best = self.config["n_estimators"]
            preds = model.predict_proba(X_val)
            score = log_loss(y_val, preds)
            cv_scores.append((score, best))
            print(f"  Fold {fold+1}: log_loss={score:.4f}, best_iter={best}")

        avg_trees = int(np.mean([b for _, b in cv_scores]))
        print(f"XGBoost: CV avg log_loss={np.mean([s for s,_ in cv_scores]):.4f}, "
              f"using n_estimators={avg_trees}")

        # Retrain on full data
        cfg_full = {k: v for k, v in self.config.items()
                    if k not in ("early_stopping_rounds",)}
        cfg_full["n_estimators"] = avg_trees
        self.model = xgb.XGBClassifier(**cfg_full, use_label_encoder=False)
        self.model.fit(X_scaled, y, verbose=False)
        self._best_ntree = avg_trees
        return self

    def predict_proba(self, row: dict) -> tuple[float, float, float]:
        """Returns (p_win, p_draw, p_loss) for a single match."""
        df = pd.DataFrame([row])
        X = self._prep_X(df)
        X_scaled = self.scaler.transform(X)
        probs = self.model.predict_proba(X_scaled)[0]
        return float(probs[0]), float(probs[1]), float(probs[2])

    def feature_importance(self) -> pd.DataFrame:
        imp = self.model.feature_importances_
        cols = [f for f in self.feature_names
                if f in [self.feature_names[i] for i in range(len(imp))]]
        return pd.DataFrame({
            "feature": self.feature_names[:len(imp)],
            "importance": imp
        }).sort_values("importance", ascending=False)

    def save(self, path: Path = None):
        path = path or MODELS_DIR / "xgboost_model.pkl"
        joblib.dump({"model": self.model, "scaler": self.scaler,
                     "features": self.feature_names}, path)
        print(f"XGBoost saved to {path}")

    def load(self, path: Path = None):
        path = path or MODELS_DIR / "xgboost_model.pkl"
        obj = joblib.load(path)
        self.model, self.scaler, self.feature_names = \
            obj["model"], obj["scaler"], obj["features"]
        return self


# ── PyTorch MLP ────────────────────────────────────────────────────────────────
class MatchMLP(nn.Module):
    """
    Multi-layer perceptron: input_dim → 64 → 32 → 16 → 3 (softmax)
    Dropout for regularisation. BatchNorm for stable training.
    """
    def __init__(self, input_dim: int, hidden_dims: list[int] = None,
                 dropout: float = 0.3):
        super().__init__()
        dims = [input_dim] + (hidden_dims or [64, 32, 16])
        layers = []
        for i in range(len(dims) - 1):
            layers.extend([
                nn.Linear(dims[i], dims[i+1]),
                nn.BatchNorm1d(dims[i+1]),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
        layers.append(nn.Linear(dims[-1], 3))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # raw logits; use CrossEntropyLoss


class NeuralNetMatchModel:
    """
    PyTorch MLP wrapper exposing the same interface as XGBoostMatchModel.
    GPU-aware (uses CUDA if available).
    """

    def __init__(self, config: dict = None, feature_names: list[str] = None):
        self.config = {**NEURAL_NET_CONFIG, **(config or {})}
        self.feature_names = feature_names or ALL_FEATURES
        self.scaler = StandardScaler()
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"NeuralNet: using device={self.device}")

    def _prep_X(self, df: pd.DataFrame) -> np.ndarray:
        cols = [f for f in self.feature_names if f in df.columns]
        return df[cols].fillna(0.0).values.astype(float)

    def fit(self, df: pd.DataFrame) -> "NeuralNetMatchModel":
        X = self._prep_X(df)
        y = np.array([encode_result(r["home_score"], r["away_score"])
                       for _, r in df.iterrows()])

        X_scaled = self.scaler.fit_transform(X)
        input_dim = X_scaled.shape[1]

        self.model = MatchMLP(
            input_dim=input_dim,
            hidden_dims=self.config["hidden_dims"],
            dropout=self.config["dropout"]
        ).to(self.device)

        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config["lr"],
            weight_decay=self.config["weight_decay"]
        )
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5
        )

        # Train/val split 80/20
        n = len(X_scaled)
        n_train = int(0.8 * n)
        idx = np.random.permutation(n)
        tr_idx, val_idx = idx[:n_train], idx[n_train:]

        X_tr = torch.FloatTensor(X_scaled[tr_idx]).to(self.device)
        y_tr = torch.LongTensor(y[tr_idx]).to(self.device)
        X_val = torch.FloatTensor(X_scaled[val_idx]).to(self.device)
        y_val = torch.LongTensor(y[val_idx]).to(self.device)

        dataset = TensorDataset(X_tr, y_tr)
        loader = DataLoader(dataset, batch_size=self.config["batch_size"], shuffle=True)

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        print(f"NeuralNet: training on {n_train} samples, validating on {n - n_train}...")
        for epoch in range(self.config["epochs"]):
            self.model.train()
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                logits = self.model(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_logits = self.model(X_val)
                val_loss = criterion(val_logits, y_val).item()

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if epoch % 20 == 0:
                print(f"  Epoch {epoch:3d}: val_loss={val_loss:.4f} "
                      f"(best={best_val_loss:.4f}, patience={patience_counter})")

            if patience_counter >= self.config["patience"]:
                print(f"  Early stopping at epoch {epoch}")
                break

        # Restore best weights
        if best_state:
            self.model.load_state_dict(best_state)

        print(f"NeuralNet: training complete. Best val_loss={best_val_loss:.4f}")
        return self

    def predict_proba(self, row: dict) -> tuple[float, float, float]:
        df = pd.DataFrame([row])
        X = self._prep_X(df)
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        return float(probs[0]), float(probs[1]), float(probs[2])

    def save(self, path: Path = None):
        path = path or MODELS_DIR / "neural_net.pt"
        torch.save({
            "model_state": self.model.state_dict(),
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "config": self.config,
            "input_dim": self.model.net[0].in_features,
        }, path)
        print(f"NeuralNet saved to {path}")

    def load(self, path: Path = None):
        path = path or MODELS_DIR / "neural_net.pt"
        ckpt = torch.load(path, map_location=self.device)
        self.scaler = ckpt["scaler"]
        self.feature_names = ckpt["feature_names"]
        self.config = ckpt["config"]
        self.model = MatchMLP(
            input_dim=ckpt["input_dim"],
            hidden_dims=self.config["hidden_dims"],
            dropout=self.config["dropout"]
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        return self
