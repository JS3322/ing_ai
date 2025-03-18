import argparse
import sys
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size=64, output_size=1):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


def load_data(file_path, is_predict=False):
    try:
        data = pd.read_csv(file_path)
        print(f"[INFO] 데이터 로드 완료: {file_path}")
        if not is_predict and 'target' not in data.columns:
            print("[ERROR] 'target' 컬럼이 없습니다.")
            sys.exit(1)
        return data
    except Exception as e:
        print(f"[ERROR] 데이터 로드 실패: {e}")
        sys.exit(1)


def preprocess_data(data, is_predict=False):
    if is_predict:
        X = data.values
        y = None
    else:
        X = data.drop('target', axis=1).values
        y = data['target'].values.reshape(-1, 1)
    print("[INFO] 전처리 완료")
    return X, y


def custom_scale(X, mean_X=None, std_X=None):
    """
    X를 표준화하는 함수. mean_X, std_X가 주어지지 않으면
    X로부터 직접 평균과 표준편차를 계산한다.
    """
    if mean_X is None:
        mean_X = X.mean(axis=0, keepdims=True)
    if std_X is None:
        std_X = X.std(axis=0, keepdims=True)
        # std가 0인 경우 대비 (분모 0 방지)
        std_X[std_X == 0] = 1.0
    X_scaled = (X - mean_X) / std_X
    return X_scaled, mean_X, std_X


def scale_data(X, y=None):
    # y가 있을 때와 없을 때를 나누어 처리
    # X, y 모두 스케일링 (표준화)
    X_scaled, mean_X, std_X = custom_scale(X)
    scaler_info = {'X_mean': mean_X, 'X_std': std_X, 'y_mean': None, 'y_std': None}
    if y is not None:
        y_scaled, mean_y, std_y = custom_scale(y)
        scaler_info['y_mean'] = mean_y
        scaler_info['y_std'] = std_y
        print("[INFO] X, y 스케일링 완료")
        return X_scaled, y_scaled, scaler_info
    else:
        print("[INFO] X 스케일링 완료 (타겟 없음)")
        return X_scaled, None, scaler_info


def inverse_scale_y(y_scaled, y_mean, y_std):
    return y_scaled * y_std + y_mean


def create_data_loader(X, y=None, batch_size=32, shuffle=True):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    if y is not None:
        y_tensor = torch.tensor(y, dtype=torch.float32)
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    else:
        dataset = torch.utils.data.TensorDataset(X_tensor)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_model(model, criterion, optimizer, train_loader, epochs=100):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"[TRAIN] Epoch [{epoch + 1}/{epochs}] Loss: {epoch_loss / len(train_loader):.4f}")
    print("[INFO] 학습 완료")
    return model


def test_model(model, criterion, test_loader):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    avg_loss = total_loss / len(test_loader)
    print(f"[TEST] 평균 손실: {avg_loss:.4f}")
    return avg_loss


def predict(model, predict_loader, y_mean=None, y_std=None):
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in predict_loader:
            inputs = batch[0]
            outputs = model(inputs).numpy()
            # 타겟 스케일링 되었다면 복원
            if y_mean is not None and y_std is not None:
                outputs = inverse_scale_y(outputs, y_mean, y_std)
            predictions.extend(outputs.flatten().tolist())
    for i, p in enumerate(predictions, start=1):
        print(f"[PREDICT] 샘플 {i} 예측값: {p:.4f}")


def optimize_hyperparameters(X_train, y_train, X_test, y_test, epochs=50):
    hidden_sizes = [32, 64, 128]
    learning_rates = [0.01, 0.001, 0.0001]

    best_loss = float('inf')
    best_config = None
    best_model_state = None

    input_size = X_train.shape[1]
    criterion = nn.MSELoss()

    train_loader = create_data_loader(X_train, y_train, batch_size=32, shuffle=True)
    test_loader = create_data_loader(X_test, y_test, batch_size=32, shuffle=False)

    for hs in hidden_sizes:
        for lr in learning_rates:
            model = SimpleNN(input_size, hidden_size=hs)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            model = train_model(model, criterion, optimizer, train_loader, epochs=epochs)
            loss = test_model(model, criterion, test_loader)
            if loss < best_loss:
                best_loss = loss
                best_config = (hs, lr)
                best_model_state = model.state_dict()

    print(f"[OPTIMIZE] 최적의 설정: hidden_size={best_config[0]}, lr={best_config[1]}, loss={best_loss:.4f}")
    return best_config, best_loss, best_model_state


def main():
    parser = argparse.ArgumentParser(description='각 작업을 별도로 1회 실행하는 예제 + optimize (No sklearn)')
    parser.add_argument('--data_dir', type=str, required=True, help='데이터 디렉토리')
    parser.add_argument('--train_data', type=str, help='학습용 CSV 파일명')
    parser.add_argument('--test_data', type=str, help='테스트용 CSV 파일명')
    parser.add_argument('--predict_data', type=str, help='예측용 CSV 파일명(타겟 없음)')

    parser.add_argument('--preprocess', action='store_true', help='전처리만 수행')
    parser.add_argument('--scale', action='store_true', help='스케일링만 수행')
    parser.add_argument('--train', action='store_true', help='학습만 수행')
    parser.add_argument('--test', action='store_true', help='평가만 수행')
    parser.add_argument('--predict', action='store_true', help='예측만 수행')
    parser.add_argument('--optimize', action='store_true', help='하이퍼파라미터 최적화만 수행')

    parser.add_argument('--epochs', type=int, default=100, help='학습 에폭 수')
    parser.add_argument('--model_path', type=str, default='model.pth', help='모델 파라미터 저장/로딩 경로')

    args = parser.parse_args()

    # 하나의 작업만 수행
    active_tasks = sum([args.preprocess, args.scale, args.train, args.test, args.predict, args.optimize])
    if active_tasks > 1:
        print("[ERROR] 하나의 실행에서 오직 하나의 작업만 수행하도록 하세요.")
        sys.exit(1)
    elif active_tasks == 0:
        print("[ERROR] 작업을 지정하지 않았습니다.")
        sys.exit(1)

    data_dir = args.data_dir

    # 전처리만 수행
    if args.preprocess:
        if not args.train_data:
            print("[ERROR] 전처리를 위해 --train_data 필요")
            sys.exit(1)
        train_path = os.path.join(data_dir, args.train_data)
        train_data = load_data(train_path, is_predict=False)
        X, y = preprocess_data(train_data, is_predict=False)
        print("[INFO] 전처리 결과: X.shape={}, y.shape={}".format(X.shape, y.shape))
        sys.exit(0)

    # 스케일링만 수행
    if args.scale:
        if not args.train_data:
            print("[ERROR] 스케일링 위해 --train_data 필요")
            sys.exit(1)
        train_path = os.path.join(data_dir, args.train_data)
        train_data = load_data(train_path, is_predict=False)
        X, y = preprocess_data(train_data, is_predict=False)
        X_scaled, y_scaled, scaler_info = scale_data(X, y)
        print("[INFO] 스케일링 결과: X_scaled.shape={}, y_scaled.shape={}".format(X_scaled.shape, y_scaled.shape))
        sys.exit(0)

    # 학습만 수행
    if args.train:
        if not args.train_data:
            print("[ERROR] 학습 위해 --train_data 필요")
            sys.exit(1)
        train_path = os.path.join(data_dir, args.train_data)
        train_data = load_data(train_path, is_predict=False)
        X, y = preprocess_data(train_data, is_predict=False)
        # 필요 시 스케일링할 경우 여기서 구현
        input_size = X.shape[1]
        model = SimpleNN(input_size)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        train_loader = create_data_loader(X, y, batch_size=32, shuffle=True)
        model = train_model(model, criterion, optimizer, train_loader, epochs=args.epochs)
        torch.save(model.state_dict(), args.model_path)
        sys.exit(0)

    # 평가만 수행
    if args.test:
        if not args.test_data:
            print("[ERROR] 평가 위해 --test_data 필요")
            sys.exit(1)
        test_path = os.path.join(data_dir, args.test_data)
        test_data = load_data(test_path, is_predict=False)
        X_test, y_test = preprocess_data(test_data, is_predict=False)
        if not os.path.exists(args.model_path):
            print("[ERROR] 모델 파일 없음")
            sys.exit(1)
        input_size = X_test.shape[1]
        model = SimpleNN(input_size)
        model.load_state_dict(torch.load(args.model_path))
        model.eval()
        criterion = nn.MSELoss()
        test_loader = create_data_loader(X_test, y_test, batch_size=32, shuffle=False)
        test_model(model, criterion, test_loader)
        sys.exit(0)

    # 예측만 수행
    if args.predict:
        if not args.predict_data:
            print("[ERROR] 예측 위해 --predict_data 필요")
            sys.exit(1)
        predict_path = os.path.join(data_dir, args.predict_data)
        predict_data = load_data(predict_path, is_predict=True)
        X_predict = predict_data.values
        if not os.path.exists(args.model_path):
            print("[ERROR] 모델 파일 없음")
            sys.exit(1)
        input_size = X_predict.shape[1]
        model = SimpleNN(input_size)
        model.load_state_dict(torch.load(args.model_path))
        model.eval()
        predict_loader = create_data_loader(X_predict, y=None, batch_size=1, shuffle=False)
        predict(model, predict_loader)
        sys.exit(0)

    # 최적화만 수행
    if args.optimize:
        if not args.train_data or not args.test_data:
            print("[ERROR] 최적화를 위해선 --train_data, --test_data 필요")
            sys.exit(1)
        train_path = os.path.join(data_dir, args.train_data)
        test_path = os.path.join(data_dir, args.test_data)
        train_data = load_data(train_path, is_predict=False)
        test_data = load_data(test_path, is_predict=False)

        X_train, y_train = preprocess_data(train_data, is_predict=False)
        X_test, y_test = preprocess_data(test_data, is_predict=False)
        best_config, best_loss, best_model_state = optimize_hyperparameters(X_train, y_train, X_test, y_test, epochs=50)
        input_size = X_train.shape[1]
        best_model = SimpleNN(input_size, hidden_size=best_config[0])
        best_model.load_state_dict(best_model_state)
        torch.save(best_model.state_dict(), args.model_path)
        print(f"[INFO] 최적 모델 저장 완료: {args.model_path}")
        sys.exit(0)


if __name__ == '__main__':
    main()
