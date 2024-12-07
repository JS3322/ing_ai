import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

# 랜덤 데이터 생성 함수
def generate_random_data(num_samples):
    x = np.random.uniform(0, 10, size=(num_samples, 1))          # x: 1의 자리
    y = np.random.uniform(0, 100, size=(num_samples, 1))         # y: 2의 자리
    z = np.random.uniform(0, 1000, size=(num_samples, 1))        # z: 3의 자리
    a = np.random.uniform(0, 10000, size=(num_samples, 1))       # a: 4의 자리
    b = np.random.uniform(0, 100000, size=(num_samples, 1))      # b: 5의 자리
    c = np.random.uniform(0, 1000000, size=(num_samples, 1))     # c: 6의 자리
    d = np.random.uniform(0, 10000000, size=(num_samples, 1))    # d: 7의 자리

    # 특성들을 하나의 배열로 합치기
    X = np.hstack((x, y, z, a, b, c, d))
    return X

# 타겟 변수 생성 함수
def generate_target(X):
    # 예시로 각 특성의 합을 타겟 값으로 설정
    y = np.sum(X, axis=1, keepdims=True)
    return y

# 모델 정의
class SimpleModel(nn.Module):
    def __init__(self, input_size=7, hidden_size=64):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# 모델 생성 및 훈련 함수
def create_model():
    # 데이터 생성
    num_samples = 1000
    X = generate_random_data(num_samples)
    y = generate_target(X)

    # 텐서로 변환
    X_tensor = torch.from_numpy(X).float()
    y_tensor = torch.from_numpy(y).float()

    # 데이터셋 및 데이터로더 생성
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    # 모델 인스턴스 생성
    model = SimpleModel()

    # 손실 함수 및 옵티마이저 정의
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 모델 훈련
    num_epochs = 100
    for epoch in range(num_epochs):
        for batch_X, batch_y in dataloader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # 모델 저장 (현재 디렉토리에 저장)
    model_save_path = os.path.join(os.path.dirname(__file__), 'model.pth')
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

# 예측 함수
def predict(input_data):
    # 입력 데이터는 (num_samples, 7)의 numpy 배열이어야 합니다.

    # 모델 불러오기
    model = SimpleModel()
    model_load_path = os.path.join(os.path.dirname(__file__), 'model.pth')
    model.load_state_dict(torch.load(model_load_path))
    model.eval()

    with torch.no_grad():
        input_tensor = torch.from_numpy(input_data).float()
        outputs = model(input_tensor)
    # 출력 값을 numpy 배열로 변환
    outputs = outputs.numpy()
    return outputs

# 예시 실행
if __name__ == "__main__":
    # 모델 생성 및 훈련
    create_model()

    # 테스트 데이터 생성
    test_X = generate_random_data(5)  # 5개의 샘플 생성
    test_y = generate_target(test_X)

    # 예측 수행
    predictions = predict(test_X)

    # 결과 출력
    for i in range(len(test_X)):
        print(f'입력: {test_X[i]}')
        print(f'실제 값: {test_y[i][0]:.2f}, 예측 값: {predictions[i][0]:.2f}\n')
