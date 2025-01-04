import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os


###############################################################################
# 1. 데이터셋 정의
###############################################################################
class SimulationDataset(Dataset):
    def __init__(self, X, y):
        # X: (N, input_dim), y: (N, output_dim)
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


###############################################################################
# 2. MLP 모델 정의
###############################################################################
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


###############################################################################
# 3. 학습용 함수들 (train_step, evaluate, train_loop)
###############################################################################
def train_step(model, optimizer, criterion, X_batch, y_batch):
    optimizer.zero_grad()
    preds = model(X_batch)
    loss = criterion(preds, y_batch)
    loss.backward()
    optimizer.step()
    return loss.item()


def evaluate(model, criterion, data_loader, device='cpu'):
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            batch_size = y_batch.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    avg_loss = total_loss / total_samples
    return avg_loss


def train_loop(model, train_loader, val_loader, epochs=50, lr=1e-3, device='cpu'):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            loss_val = train_step(model, optimizer, criterion, X_batch, y_batch)
            running_loss += loss_val

        val_loss = evaluate(model, criterion, val_loader, device=device)

        if epoch % 10 == 0 or epoch == epochs:
            print(f"[Epoch {epoch:3d}/{epochs}] "
                  f"Train Loss: {running_loss / len(train_loader):.4f}, "
                  f"Val Loss: {val_loss:.4f}")
    return model


###############################################################################
# 4. 데이터 로드 (LF & HF)
###############################################################################
def load_sim_data(lf_path='lf_data.csv', hf_path='hf_data.csv',
                  input_cols=None, output_cols=None):
    """
    - lf_data.csv (저정밀, 샘플 수 많음)
    - hf_data.csv (고정밀, 샘플 수 적음)
    - input_cols: 공정 파라미터나 기타 입력 피처
    - output_cols: 시뮬레이션 결과(예: 소자 특성)
    """
    if input_cols is None or output_cols is None:
        raise ValueError("input_cols, output_cols must be specified.")

    # 가정: CSV 파일에 input_cols + output_cols가 모두 존재
    lf_df = pd.read_csv(lf_path)
    hf_df = pd.read_csv(hf_path)

    X_lf = lf_df[input_cols].values.astype(np.float32)
    y_lf = lf_df[output_cols].values.astype(np.float32)

    X_hf = hf_df[input_cols].values.astype(np.float32)
    y_hf = hf_df[output_cols].values.astype(np.float32)

    return X_lf, y_lf, X_hf, y_hf


###############################################################################
# 5. Transfer Learning (LF → HF Fine-tuning) 예시
###############################################################################
def transfer_learning_example(
        lf_path='lf_data.csv', hf_path='hf_data.csv',
        input_cols=['param1', 'param2', 'param3'],
        output_cols=['target1'],  # 회귀 예시
        device='cpu'
):
    # 1) 데이터 로드
    X_lf, y_lf, X_hf, y_hf = load_sim_data(lf_path, hf_path, input_cols, output_cols)
    print("[LF data] shape =", X_lf.shape, y_lf.shape)
    print("[HF data] shape =", X_hf.shape, y_hf.shape)

    # 분할 (예시: LF 데이터를 train/val으로 나눔)
    # HF 데이터도 train/val 또는 cross-validation 고려 가능
    num_lf = len(X_lf)
    idxs = np.arange(num_lf)
    np.random.shuffle(idxs)
    train_ratio = 0.8
    train_size = int(num_lf * train_ratio)

    train_idx = idxs[:train_size]
    val_idx = idxs[train_size:]

    X_lf_train, y_lf_train = X_lf[train_idx], y_lf[train_idx]
    X_lf_val, y_lf_val = X_lf[val_idx], y_lf[val_idx]

    # HF는 샘플이 적다고 가정 → 전부 train으로 쓰거나, 소량만 val
    X_hf_train, y_hf_train = X_hf, y_hf  # 가정

    # 2) Dataset & DataLoader
    lf_train_dataset = SimulationDataset(X_lf_train, y_lf_train)
    lf_val_dataset = SimulationDataset(X_lf_val, y_lf_val)
    hf_train_dataset = SimulationDataset(X_hf_train, y_hf_train)

    lf_train_loader = DataLoader(lf_train_dataset, batch_size=32, shuffle=True)
    lf_val_loader = DataLoader(lf_val_dataset, batch_size=32, shuffle=False)
    hf_train_loader = DataLoader(hf_train_dataset, batch_size=8, shuffle=True)
    # HF 데이터 수 적으니 batch_size도 작게

    # 3) Pre-training with LF
    print("\n=== [Pre-training with LF data] ===")
    input_dim = len(input_cols)
    output_dim = len(output_cols)
    base_model = SimpleMLP(input_dim=input_dim, hidden_dim=64, output_dim=output_dim)
    base_model = train_loop(base_model, lf_train_loader, lf_val_loader,
                            epochs=50, lr=1e-3, device=device)

    # 4) Fine-tuning with HF
    print("\n=== [Fine-tuning with HF data] ===")
    # 일반적으로는 base_model.weight 파라미터를 가져와서 다시 학습
    # 여기서는 base_model 자체를 그대로 활용
    finetune_model = base_model

    # Fine-tuning 시에는 학습률을 더 낮추거나 일부 레이어만 학습하기도 함
    criterion = nn.MSELoss()
    optimizer = optim.Adam(finetune_model.parameters(), lr=1e-4)

    finetune_model.to(device)
    for epoch in range(1, 31):
        finetune_model.train()
        running_loss = 0.0
        for X_batch, y_batch in hf_train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            preds = finetune_model(X_batch)
            loss_val = criterion(preds, y_batch)
            loss_val.backward()
            optimizer.step()
            running_loss += loss_val.item()

        if epoch % 5 == 0 or epoch == 30:
            avg_train_loss = running_loss / len(hf_train_loader)
            print(f"[Fine-tune Epoch {epoch:2d}/30] HF Train Loss={avg_train_loss:.4f}")

    print("\n=== [Transfer Learning Done] ===")

    # 만약 HF용 val 데이터가 별도로 있다면, 성능 검증을 수행
    # 예) hf_val_loader로 evaluate()

    return finetune_model


###############################################################################
# 6. 실행 예시
###############################################################################
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)

    # 예시 CSV 경로 (사용자 환경에 맞게 바꿔주세요)
    lf_path = 'low_fidelity_data.csv'
    hf_path = 'high_fidelity_data.csv'

    # (param1, param2, param3) -> (target1)
    # 이 예시에서는 input 3, output 1
    model = transfer_learning_example(
        lf_path=lf_path,
        hf_path=hf_path,
        input_cols=['param1', 'param2', 'param3'],
        output_cols=['target1'],
        device=device
    )

    # 추가 활용
    # model.predict(...) 등으로 Surrogate Model로 활용 가능


if __name__ == "__main__":
    main()
