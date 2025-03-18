import os
import tensorflow as tf
import pandas as pd


class HBMBandwidthModel:
    def __init__(self, train_csv='reference/train.csv', test_csv='reference/test.csv'):
        # 현재 스크립트의 절대 경로를 기준으로 프로젝트 루트 구하기 (프로젝트의 절대 경로 필요 검토와 각 스텝을 모듈화 고려)
        # __file__ 변수가 없으면 os.getcwd()를 사용하세요.
        self.project_root = os.path.abspath(os.path.dirname(__file__))
        self.train_csv_path = os.path.join(train_csv)
        self.test_csv_path = os.path.join(test_csv)
        self.model = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.X_min = None
        self.X_max = None

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
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(5,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1)  # 회귀 문제이므로 단일 출력
        ])

        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                           loss='mse',
                           metrics=['mae'])
        print("모델 생성 완료.")

    def train_model(self, epochs=100, batch_size=32):
        """
        학습 데이터를 이용하여 모델을 학습합니다.
        EarlyStopping을 활용하여 검증 손실이 개선되지 않으면 학습을 중단합니다.
        """
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        history = self.model.fit(self.X_train, self.y_train,
                                 validation_split=0.2,
                                 epochs=epochs,
                                 batch_size=batch_size,
                                 callbacks=[early_stop],
                                 verbose=1)
        print("모델 학습 완료.")
        return history

    def evaluate_model(self):
        """
        테스트 데이터를 이용하여 모델 성능을 평가합니다.
        """
        loss, mae = self.model.evaluate(self.X_test, self.y_test, verbose=2)
        print(f"테스트 손실(MSE): {loss:.4f}, 테스트 MAE: {mae:.4f}")
        return loss, mae

    def predict(self, X):
        """
        입력 데이터를 받아 예측 결과를 반환합니다.
        (입력 데이터는 정규화된 상태여야 합니다.)
        """
        return self.model.predict(X)

    def save_model(self, save_dir='saved_model', filename='model.h5'):
        """
        현재 학습된 모델을 지정된 디렉토리에 HDF5 형식으로 저장합니다.
        """
        # project_root 또는 절대 경로 필요?
        save_path = os.path.join(self.project_root, save_dir)
        os.makedirs(save_path, exist_ok=True)
        full_save_path = os.path.join(save_path, filename)
        # HDF5 형식으로 모델 저장 (.h5 확장자 사용)
        self.model.save(full_save_path)
        print(f"모델이 '{full_save_path}'에 저장되었습니다.")


def check_gpu():
    """
    TensorFlow를 이용해 GPU 사용 가능 여부를 확인합니다.
    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"사용 가능한 GPU: {[gpu.name for gpu in gpus]}")
    else:
        print("GPU가 사용 가능하지 않습니다.")


# 클래스 인스턴스 생성, 데이터 로드, 모델 생성, 학습, 평가, 저장 예제
