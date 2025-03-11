import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# CSV 파일 경로
train_csv = os.path.join('reference', 'train.csv')
test_csv = os.path.join('reference', 'test.csv')

# 데이터 로드 (pandas DataFrame 사용)
train_df = pd.read_csv(train_csv)
test_df = pd.read_csv(test_csv)

# 입력 특성과 타깃 변수 분리
feature_columns = ['layer_distance', 'interconnect_width', 'material_resistivity', 'dielectric_constant', 'temperature']
target_column = 'bandwidth'

X_train = train_df[feature_columns].values
y_train = train_df[target_column].values
X_test = test_df[feature_columns].values
y_test = test_df[target_column].values

# 데이터 정규화 (Min-Max scaling 예시)
# 학습 데이터의 최소, 최대값을 이용하여 정규화 (추후 예측 시 동일 스케일 사용 필요)
X_min = X_train.min(axis=0)
X_max = X_train.max(axis=0)
X_train_norm = (X_train - X_min) / (X_max - X_min)
X_test_norm = (X_test - X_min) / (X_max - X_min)

# 타깃은 단순 회귀이므로 별도의 스케일링 없이 사용 (필요시 scaling 가능)

# 모델 구성: 간단한 완전 연결 신경망 (Dense layers)로 회귀 모델 구축
model = models.Sequential([
    layers.Input(shape=(len(feature_columns),)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(1)  # 출력: 대역폭 (회귀 문제)
])

# 모델 컴파일
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='mse',
              metrics=['mae'])

# 학습 진행 (EarlyStopping 콜백 사용)
early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(X_train_norm, y_train,
                    validation_split=0.2,
                    epochs=100,
                    batch_size=32,
                    callbacks=[early_stop],
                    verbose=1)

# 모델 평가
test_loss, test_mae = model.evaluate(X_test_norm, y_test, verbose=2)
print(f'\nTest MSE: {test_loss:.4f}, Test MAE: {test_mae:.4f}')

# 예측 예제
predictions = model.predict(X_test_norm)
print("예측 예시:", predictions[:5].flatten())