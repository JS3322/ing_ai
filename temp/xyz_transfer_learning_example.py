from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense, Input
import pandas as pd

def transfer_learning(csv_path, base_model_path, new_model_path):
    """
    전이학습을 수행하는 함수

    Args:
        csv_path (str): 전이학습용 데이터가 저장된 CSV 경로
        base_model_path (str): 기존 모델 파일 경로 (.h5)
        new_model_path (str): 전이학습 후 저장할 모델 파일 경로 (.h5)
    """
    # CSV 파일 읽기
    data = pd.read_csv(csv_path)
    X = data[["x", "y", "z"]]
    y = data["multipl_sum"]
    
    # 기존 모델 로드
    base_model = load_model(base_model_path)
    
    # 기존 모델의 가중치를 고정 (전이학습에서 일부 레이어를 고정할 때 사용)
    for layer in base_model.layers:
        layer.trainable = False
    
    # 새로운 출력 레이어 추가
    input_layer = Input(shape=(3,))
    x = base_model(input_layer)
    output_layer = Dense(1, activation="linear", name="new_output")(x)
    
    # 새로운 모델 구성
    new_model = Model(inputs=input_layer, outputs=output_layer)
    new_model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    
    # 전이학습 수행
    new_model.fit(X, y, epochs=10, batch_size=2, verbose=1)
    
    # 새로운 모델 저장
    new_model.save(new_model_path)
    print(f"새로운 모델이 {new_model_path}에 저장되었습니다.")

# 함수 실행 예시
csv_path = "transfer_learning_data.csv"
base_model_path = "base_model.h5"  # 기존 모델 파일 경로
new_model_path = "new_model.h5"    # 새로운 모델 파일 저장 경로

transfer_learning(csv_path, base_model_path, new_model_path)