import os
import tensorflow as tf
import pandas as pd

# 재현성을 위한 시드 설정
tf.random.set_seed(42)

# 데이터 샘플 수
n_samples = 1000

# 문헌 기반 파라미터 범위에 따른 합성 데이터 생성 (TensorFlow 연산 사용)
# 층 간 거리: 5 ~ 20 µm
layer_distance = tf.random.uniform([n_samples], minval=5, maxval=20, dtype=tf.float32)
# 인터커넥트 폭: 0.5 ~ 5 µm
interconnect_width = tf.random.uniform([n_samples], minval=0.5, maxval=5, dtype=tf.float32)
# 재료 저항: 1 ~ 5 Ω·cm (효과적인 값)
material_resistivity = tf.random.uniform([n_samples], minval=1, maxval=5, dtype=tf.float32)
# 유전율: 2 ~ 10 (예: SiO2 ~ Low-k)
dielectric_constant = tf.random.uniform([n_samples], minval=2, maxval=10, dtype=tf.float32)
# 온도: 25 ~ 85 °C
temperature = tf.random.uniform([n_samples], minval=25, maxval=85, dtype=tf.float32)

# 실제 환경의 불확실성을 반영할 노이즈 (정규분포)
noise = tf.random.normal([n_samples], mean=0.0, stddev=0.05, dtype=tf.float32)

# TensorFlow 연산을 통한 대역폭 계산
# 예시 수식: 
# bandwidth = (인터커넥트 폭 / 층 간 거리) / (재료 저항 * 유전율) * ((100 - 온도)/100) + 노이즈
base_bandwidth = (interconnect_width / layer_distance) / (material_resistivity * dielectric_constant)
bandwidth = base_bandwidth * ((100 - temperature) / 100) + noise

# 음수 값은 0으로 클리핑
bandwidth = tf.maximum(bandwidth, 0)

# 텐서 데이터를 NumPy 배열로 변환 후, 판다스 DataFrame 생성
data = {
    'layer_distance': layer_distance.numpy(),
    'interconnect_width': interconnect_width.numpy(),
    'material_resistivity': material_resistivity.numpy(),
    'dielectric_constant': dielectric_constant.numpy(),
    'temperature': temperature.numpy(),
    'bandwidth': bandwidth.numpy()
}
df = pd.DataFrame(data)

# 데이터 셔플 및 학습/테스트 데이터 수동 분리 (80% train, 20% test)
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
train_size = int(0.8 * n_samples)
train_df = df_shuffled.iloc[:train_size]
test_df = df_shuffled.iloc[train_size:]

# 파일 저장 디렉토리 생성
output_dir = 'reference'
os.makedirs(output_dir, exist_ok=True)

# 학습 및 테스트 데이터 CSV 파일로 저장
train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)

print("TensorFlow 기반 합성 데이터가 'reference/train.csv'와 'reference/test.csv'로 저장되었습니다.")