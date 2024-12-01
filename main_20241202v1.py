import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

# 컬럼 리스트와 데이터 행 수 정의
column_list = ['a', 'b', 'c']  # 예시 컬럼명
data_row_count = 1000          # 생성할 데이터 행 수

# 랜덤 데이터 생성 함수
def generate_random_data(column_list, data_row_count):
    num_columns = len(column_list)
    X = np.zeros((data_row_count, num_columns), dtype=np.float32)

    for idx, col in enumerate(column_list):
        # 각 컬럼의 인덱스에 따라 데이터 범위 설정
        lower_bound = 0
        upper_bound = 10 ** (idx + 1)  # 예: a=0-9, b=0-99, c=0-999 등
        X[:, idx] = np.random.uniform(lower_bound, upper_bound, data_row_count)
    
    return X

# 타겟 변수 생성 함수
def generate_target(X):
    # 예시: 각 특성의 합을 타겟으로 설정
    y = np.sum(X, axis=1, keepdims=True)
    return y

# 모델 정의
class SimpleModel(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# 모델 생성, 훈련 및 저장 함수
def create_model_and_save(column_list, data_row_count):
    # 데이터 생성
    X = generate_random_data(column_list, data_row_count)
    y = generate_target(X)

    # 텐서로 변환
    X_tensor = torch.from_numpy(X)
    y_tensor = torch.from_numpy(y)

    # 데이터셋 및 데이터로더 생성
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    # 모델 인스턴스 생성
    input_size = len(column_list)
    model = SimpleModel(input_size=input_size)

    # 손실 함수 및 옵티마이저 정의
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 모델 훈련
    num_epochs = 100
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_X, batch_y in dataloader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            avg_loss = epoch_loss / len(dataloader)
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

    # 모델 저장 (현재 스크립트가 위치한 디렉토리에 저장)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_save_path = os.path.join(script_dir, 'model.pth')

    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

# 모델 로드 및 예측 함수
def load_model_and_predict(column_list, input_data):
    # 모델 아키텍처 정의
    input_size = len(column_list)
    model = SimpleModel(input_size=input_size)

    # 모델 로드 (현재 스크립트가 위치한 디렉토리에서 'model.pth' 파일 읽기)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_load_path = os.path.join(script_dir, 'model.pth')

    if not os.path.exists(model_load_path):
        raise FileNotFoundError(f"Model file not found at {model_load_path}")

    model.load_state_dict(torch.load(model_load_path))
    model.eval()

    # 입력 데이터를 torch.Tensor로 변환
    input_tensor = torch.from_numpy(input_data).float()
    # 필요한 경우 배치 차원 추가
    if input_tensor.dim() == 1:
        input_tensor = input_tensor.unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)

    # 출력 값을 numpy 배열로 변환
    outputs = outputs.numpy()
    return outputs

# 예시 실행
if __name__ == "__main__":
    # 모델 생성, 훈련 및 저장
    create_model_and_save(column_list, data_row_count)

    # 테스트 데이터 생성
    test_X = generate_random_data(column_list, 5)  # 5개의 샘플 생성
    test_y = generate_target(test_X)

    # 예측 수행
    predictions = load_model_and_predict(column_list, test_X)

    # 결과 출력
    for i in range(len(test_X)):
        print(f'입력: {test_X[i]}')
        print(f'실제 값: {test_y[i][0]:.2f}, 예측 값: {predictions[i][0]:.2f}\n')