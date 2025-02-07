import sys
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def main():
    # 인자 개수 체크
    if len(sys.argv) < 4:
        print("Usage: python script.py <model_path> <csv_path> <feature_columns>")
        print("예: python script.py model.h5 data.csv x,y,z")
        sys.exit(1)
    
    # 커맨드라인 인자 처리
    model_path = sys.argv[1]
    csv_path = sys.argv[2]
    feature_cols = sys.argv[3].split(",")  # 예: "x,y,z" -> ['x', 'y', 'z']
    
    # 1. TensorFlow 모델 로드 (tf.keras.models.load_model 사용)
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"TensorFlow 모델을 '{model_path}'에서 성공적으로 로드했습니다.")
    except Exception as e:
        print("모델 로드 중 오류 발생:", e)
        sys.exit(1)
    
    # 2. CSV 파일 로드 후 지정한 피처 컬럼 선택
    try:
        df = pd.read_csv(csv_path)
        print(f"CSV 파일 '{csv_path}'를 로드했습니다. 총 샘플 수: {len(df)}")
    except Exception as e:
        print("CSV 파일 로드 실패:", e)
        sys.exit(1)
    
    # 선택한 피처 컬럼들이 CSV에 모두 존재하는지 확인
    for col in feature_cols:
        if col not in df.columns:
            print(f"오류: '{col}' 컬럼이 CSV 파일에 존재하지 않습니다.")
            sys.exit(1)
    
    # 입력 데이터 추출 및 float32 형식으로 변환
    X = df[feature_cols].values.astype(np.float32)
    
    # 3. GradientTape를 사용하여 모델 예측에 대한 입력 기울기(gradient) 계산
    X_tensor = tf.convert_to_tensor(X)
    with tf.GradientTape() as tape:
        tape.watch(X_tensor)
        predictions = model(X_tensor)
    grads = tape.gradient(predictions, X_tensor)
    
    if grads is None:
        print("Gradient 계산에 실패하였습니다.")
        sys.exit(1)
    
    # 4. 각 피처별로 절대값 기울기의 평균을 계산하여 피처 중요도 산출
    grads_np = grads.numpy()
    # grads_np.shape: (num_samples, num_features)
    feature_importance = np.mean(np.abs(grads_np), axis=0)
    
    # 5. 결과를 막대그래프로 시각화하고 이미지 파일로 저장
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(feature_importance)), feature_importance, color="skyblue")
    plt.xticks(range(len(feature_importance)), feature_cols, rotation=45)
    plt.xlabel("Feature")
    plt.ylabel("Mean Absolute Gradient")
    plt.title("Feature Importance based on Mean Absolute Gradients")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    plt.show()
    print("피처 영향도 차트가 'feature_importance.png'로 저장되었습니다.")

if __name__ == "__main__":
    main()