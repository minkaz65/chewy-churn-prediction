"""
PyTorch churn model for the Chewy project.
Run:  python train_pytorch.py  (expects ../data/chewy_features.parquet or the CSV)
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report

SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)


def load_features():
    try:
        return pd.read_parquet("../data/chewy_features.parquet")
    except Exception:
        return pd.read_csv("../data/chewy_customers.csv").drop_duplicates("customer_id")


def prepare(df):
    cat_cols = [c for c in ["pet_type", "region", "plan", "primary_category"] if c in df]
    X = pd.get_dummies(df.drop(columns=["churned", "customer_id"]), columns=cat_cols, dtype=float)
    X = X.fillna(X.median(numeric_only=True))
    y = df["churned"].values.astype(np.float32)
    X_tmp, X_te, y_tmp, y_te = train_test_split(X.values, y, test_size=0.15, stratify=y, random_state=SEED)
    X_tr, X_va, y_tr, y_va = train_test_split(X_tmp, y_tmp, test_size=0.1765, stratify=y_tmp, random_state=SEED)
    sc = StandardScaler().fit(X_tr)
    return [sc.transform(a).astype(np.float32) for a in (X_tr, X_va, X_te)], (y_tr, y_va, y_te)


class ChurnMLP(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.BatchNorm1d(64), nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def main():
    (X_tr, X_va, X_te), (y_tr, y_va, y_te) = prepare(load_features())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pos_weight = (len(y_tr) - y_tr.sum()) / y_tr.sum()

    model = ChurnMLP(X_tr.shape[1]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    dl = DataLoader(TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)), batch_size=512, shuffle=True)
    Xv, yv = torch.tensor(X_va).to(device), torch.tensor(y_va).to(device)

    best_auc, best_state, bad = 0.0, None, 0
    for epoch in range(60):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            criterion(model(xb), yb).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            auc = roc_auc_score(y_va, torch.sigmoid(model(Xv)).cpu().numpy())
        if auc > best_auc:
            best_auc, best_state, bad = auc, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 8:
                print(f"Early stop at epoch {epoch + 1}")
                break
    model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        proba = torch.sigmoid(model(torch.tensor(X_te).to(device))).cpu().numpy()
    print(f"Test ROC-AUC: {roc_auc_score(y_te, proba):.4f} | PR-AUC: {average_precision_score(y_te, proba):.4f}")
    print(classification_report(y_te, (proba >= 0.5).astype(int), target_names=["Retained", "Churned"]))
    torch.save(model.state_dict(), "../models/chewy_churn_pytorch.pt")
    print("Saved ../models/chewy_churn_pytorch.pt")


if __name__ == "__main__":
    main()
