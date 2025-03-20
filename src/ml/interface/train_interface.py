import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from datetime import datetime


class HBMNet(nn.Module):
    def __init__(self):
        super(HBMNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    
    def forward(self, x):
        return self.layers(x)


class HBMDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class HBMBandwidthModel:
    def __init__(self, data_path='reference', model_path=None):
        # 현재 스크립트의 절대 경로를 기준으로 프로젝트 루트 구하기
        self.project_root = os.path.abspath(os.path.dirname(__file__))
        self.data_path = data_path
        self.model_path = model_path if model_path else data_path
        self.train_csv_path = os.path.join(self.data_path, 'train.csv')
        self.test_csv_path = os.path.join(self.data_path, 'test.csv')
        self.model = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.X_min = None
        self.X_max = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def load_data(self):
        """
        CSV 파일에서 데이터를 로드하고, 입력 특성을 정규화합니다.
        """
        # CSV 파일 로드
        train_df = pd.read_csv(self.train_csv_path)
        test_df = pd.read_csv(self.test_csv_path)

        # 입력 특성과 타깃 변수 지정
        feature_columns = ['layer_distance', 'interconnect_width', 'material_resistivity', 'dielectric_constant',
                           'temperature']
        target_column = 'bandwidth'
        self.X_train = train_df[feature_columns].values
        self.y_train = train_df[target_column].values
        self.X_test = test_df[feature_columns].values
        self.y_test = test_df[target_column].values

        # Min-Max 정규화 (학습 데이터의 최소/최대값 사용)
        self.X_min = self.X_train.min(axis=0)
        self.X_max = self.X_train.max(axis=0)
        self.X_train = (self.X_train - self.X_min) / (self.X_max - self.X_min)
        self.X_test = (self.X_test - self.X_min) / (self.X_max - self.X_min)

        print("데이터 로드 및 정규화 완료.")

    def create_model(self):
        """
        입력층, 은닉층, 출력층을 포함하는 간단한 회귀 모델을 생성합니다.
        """
        self.model = HBMNet().to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        print("모델 생성 완료.")

    def train_model(self, epochs=100, batch_size=32):
        """
        학습 데이터를 이용하여 모델을 학습합니다.
        EarlyStopping을 활용하여 검증 손실이 개선되지 않으면 학습을 중단합니다.
        """
        # 데이터셋 및 데이터로더 생성
        train_dataset = HBMDataset(self.X_train, self.y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # 검증 데이터셋 생성 (20% 분할)
        val_size = int(0.2 * len(train_dataset))
        train_size = len(train_dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Early stopping 설정
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        best_model_state = None
        
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            # 학습
            self.model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            # 검증
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    outputs = self.model(X_batch)
                    val_loss += self.criterion(outputs, y_batch).item()
            
            # 손실 기록
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            
            # Early stopping 체크
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 최고 성능 모델 상태 저장
                best_model_state = {
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epoch': epoch + 1
                }
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'Early stopping triggered after {epoch+1} epochs')
                    # Early stopping 시 최고 성능 모델 복원
                    if best_model_state is not None:
                        self.model.load_state_dict(best_model_state['model_state_dict'])
                        self.optimizer.load_state_dict(best_model_state['optimizer_state_dict'])
                        print(f"최고 성능 모델 (epoch {best_model_state['epoch']}) 복원됨")
                    return history
        
        # 모든 에포크가 완료된 경우 최고 성능 모델 복원
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state['model_state_dict'])
            self.optimizer.load_state_dict(best_model_state['optimizer_state_dict'])
            print(f"최고 성능 모델 (epoch {best_model_state['epoch']}) 복원됨")
        
        print("모델 학습 완료.")
        return history

    def evaluate_model(self):
        """
        테스트 데이터를 이용하여 모델 성능을 평가합니다.
        """
        self.model.eval()
        test_dataset = HBMDataset(self.X_test, self.y_test)
        test_loader = DataLoader(test_dataset, batch_size=32)
        
        total_loss = 0
        total_mae = 0
        n_batches = 0
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                outputs = self.model(X_batch)
                
                # MSE 계산
                loss = self.criterion(outputs, y_batch)
                total_loss += loss.item()
                
                # MAE 계산
                mae = torch.mean(torch.abs(outputs - y_batch))
                total_mae += mae.item()
                
                n_batches += 1
        
        avg_loss = total_loss / n_batches
        avg_mae = total_mae / n_batches
        
        print(f"테스트 손실(MSE): {avg_loss:.4f}, 테스트 MAE: {avg_mae:.4f}")
        return avg_loss, avg_mae

    def predict(self, X):
        """
        입력 데이터를 받아 예측 결과를 반환합니다.
        (입력 데이터는 정규화된 상태여야 합니다.)
        """
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            predictions = self.model(X_tensor)
        return predictions.cpu().numpy()

    def save_model(self, filename='model.pth'):
        """
        현재 학습된 모델을 타임스탬프가 포함된 이름으로 저장하고 model.pth 심볼릭 링크를 생성합니다.
        """
        os.makedirs(self.model_path, exist_ok=True)
        
        # 타임스탬프로 된 모델 파일명 생성 (YYYYMMDD_HHMMSS.pth)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        timestamped_filename = f"{timestamp}.pth"
        timestamped_path = os.path.join(self.model_path, timestamped_filename)
        
        # 모델 저장
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'X_min': self.X_min,
            'X_max': self.X_max
        }, timestamped_path)
        print(f"모델이 '{timestamped_path}'에 저장되었습니다.")
        
        # model.pth 심볼릭 링크 생성
        link_path = os.path.join(self.model_path, filename)
        if os.path.exists(link_path):
            os.remove(link_path)  # 기존 링크가 있다면 제거
        os.symlink(timestamped_filename, link_path)
        print(f"심볼릭 링크 '{link_path}'가 생성되었습니다.")


def check_gpu():
    """
    PyTorch를 이용해 GPU 사용 가능 여부를 확인합니다.
    """
    if torch.cuda.is_available():
        print(f"사용 가능한 GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU가 사용 가능하지 않습니다.")


# 클래스 인스턴스 생성, 데이터 로드, 모델 생성, 학습, 평가, 저장 예제
