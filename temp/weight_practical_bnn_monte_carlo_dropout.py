import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np


# -----------------------
# 1) 데이터셋 (예시)
# -----------------------
class SimpleDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# -----------------------
# 2) 모델 정의 (MC Dropout)
# -----------------------
class MCDropoutNet(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, output_dim=1, dropout_p=0.1):
        super(MCDropoutNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

        # Monte Carlo Dropout
        self.drop = nn.Dropout(p=dropout_p)

    def forward(self, x):
        # train 모드가 아니어도 Dropout을 적용하기 위해서
        # forward 함수에서 무조건 self.drop을 호출
        x = self.fc1(x)
        x = self.relu(x)
        x = self.drop(x)

        x = self.fc2(x)
        x = self.relu(x)
        x = self.drop(x)

        x = self.fc3(x)
        return x


# -----------------------
# 3) 학습 루프
# -----------------------
def train_mc_dropout(model, train_loader, epochs=100, lr=1e-3, device='cpu'):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        model.train()  # 학습 모드 (dropout 활성)
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds.squeeze(), y_batch.squeeze())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        if epoch % 20 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss={running_loss / len(train_loader):.4f}")


# -----------------------
# 4) 예측 시 불확실성 추정
# -----------------------
def predict_with_uncertainty(model, X_input, n_samples=50, device='cpu'):
    """
    MC Dropout:
    같은 입력(X_input)에 대해, forward()를 n번 반복 호출 -> 결과 분포 확인
    """
    model.to(device)
    model.eval()  # 일반적으로 eval()이면 dropout 비활성화되지만,
    # 여기선 forward() 안에서 dropout을 강제 적용하므로 그대로 가능.

    X_t = torch.tensor(X_input, dtype=torch.float32).to(device)
    preds_list = []

    with torch.no_grad():
        for _ in range(n_samples):
            y_out = model(X_t)  # dropout이 적용된 forward
            preds_list.append(y_out.cpu().numpy())

    preds_array = np.array(preds_list)  # shape = (n_samples, batch, output_dim)
    mean_preds = np.mean(preds_array, axis=0)  # (batch, output_dim)
    std_preds = np.std(preds_array, axis=0)  # (batch, output_dim)
    return mean_preds, std_preds


# -----------------------
# 5) 데모 실행
# -----------------------
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)

    # (예시) 간단한 1차원 데이터: y = sin(x) + noise
    np.random.seed(123)
    X = np.linspace(-3, 3, 100).reshape(-1, 1).astype(np.float32)
    y = (np.sin(X) + 0.1 * np.random.randn(*X.shape)).astype(np.float32)

    dataset = SimpleDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

    # 모델 구성
    model = MCDropoutNet(input_dim=1, hidden_dim=64, output_dim=1, dropout_p=0.1)

    # 학습
    train_mc_dropout(model, train_loader, epochs=200, lr=1e-3, device=device)

    # 예측
    X_test = np.linspace(-4, 4, 41).reshape(-1, 1).astype(np.float32)  # 범위 살짝 확대
    mean_preds, std_preds = predict_with_uncertainty(model, X_test, n_samples=100, device=device)

    # 결과 시각화 (matplotlib 필요)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 5))
    plt.scatter(X, y, c='blue', label='Train data')
    plt.plot(X_test, mean_preds, c='red', label='Predict Mean')
    plt.fill_between(X_test.squeeze(),
                     (mean_preds - 2 * std_preds).squeeze(),
                     (mean_preds + 2 * std_preds).squeeze(),
                     color='pink', alpha=0.3, label='±2std')
    plt.title("MC Dropout BNN Example")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
