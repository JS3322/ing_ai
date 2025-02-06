import sys
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# (선택 사항) 모델 로드: 로그용 또는 참고용으로 모델을 로드합니다.
def load_model(model_path):
    """
    모델 파일 경로로부터 모델을 로드합니다.
    joblib, tf.keras, torch 순으로 로드를 시도합니다.
    """
    # 1. joblib 로드 시도 (scikit-learn 등)
    try:
        model = joblib.load(model_path)
        print("joblib를 사용하여 모델 로드 성공.")
        return model
    except Exception as e:
        print("joblib 로드 실패:", e)

    # 2. Keras/TensorFlow 모델 로드 시도 (.h5 파일 등)
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(model_path)
        print("Keras/TensorFlow 모델 로드 성공.")
        return model
    except Exception as e:
        print("Keras/TensorFlow 모델 로드 실패:", e)

    # 3. PyTorch 모델 로드 시도
    try:
        import torch
        model = torch.load(model_path, map_location=torch.device('cpu'))
        model.eval()  # 평가 모드로 전환
        print("PyTorch 모델 로드 성공.")
        return model
    except Exception as e:
        print("PyTorch 모델 로드 실패:", e)
        sys.exit("모델 로드에 실패하였습니다.")

def train_surrogate_model(X, y):
    """
    입력 피처(X)와 예측값(y)를 이용하여 surrogate 모델(여기서는 XGBoost 회귀 모델)을 학습합니다.
    이 surrogate 모델은 원래 모델의 예측 결과를 근사합니다.
    """
    try:
        import xgboost as xgb
    except ImportError:
        print("xgboost가 설치되어 있지 않습니다. pip install xgboost 로 설치하세요.")
        sys.exit(1)
    
    # 회귀 문제라고 가정 (분류 문제의 경우 objective 및 평가 방법을 변경)
    surrogate = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, random_state=42)
    surrogate.fit(X, y)
    print("Surrogate 모델(XGBoost) 학습 완료.")
    return surrogate

def compute_shap_from_surrogate(surrogate, X):
    """
    surrogate 모델과 입력 데이터 X를 사용하여 SHAP 값을 계산하고 summary plot을 저장합니다.
    """
    # TreeExplainer를 사용 (XGBoost 모델에 최적화됨)
    explainer = shap.TreeExplainer(surrogate)
    shap_values = explainer.shap_values(X)
    
    # SHAP summary plot 생성 및 저장
    shap.summary_plot(shap_values, X, show=False)
    plt.savefig("shap_summary.png")
    plt.close()
    print("SHAP summary plot이 'shap_summary.png'로 저장되었습니다.")

def main():
    """
    사용법:
      python script.py <모델_파일경로> <예측결과_csv_파일경로> <피처컬럼목록> <예측값컬럼명>
    
    <피처컬럼목록>은 쉼표로 구분된 컬럼 이름 문자열로 전달합니다.
    예)
      python script.py model.h5 predict_result.csv col1,col2,col3 prediction
    """
    if len(sys.argv) < 5:
        print("사용법: python script.py <모델_파일경로> <예측결과_csv_파일경로> <피처컬럼목록> <예측값컬럼명>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    predict_csv_path = sys.argv[2]
    feature_columns = sys.argv[3].split(",")
    prediction_column = sys.argv[4]
    
    # 모델 로드 (필요시; SHAP 계산에는 surrogate 모델 사용)
    _ = load_model(model_path)
    
    # 예측 결과 CSV 파일 로드 (피처와 예측값 컬럼 포함)
    try:
        df = pd.read_csv(predict_csv_path)
        print(f"CSV 파일 '{predict_csv_path}' 로드 성공. 총 샘플 수: {len(df)}")
    except Exception as e:
        print(f"CSV 파일 로드 실패: {e}")
        sys.exit(1)
    
    # CSV에서 지정된 피처 컬럼과 예측값 컬럼 추출
    try:
        X = df[feature_columns].values
    except KeyError as e:
        print(f"CSV 파일에 지정된 피처 컬럼이 존재하지 않습니다: {e}")
        sys.exit(1)
    
    try:
        y = df[prediction_column].values
    except KeyError as e:
        print(f"CSV 파일에 지정된 예측값 컬럼이 존재하지 않습니다: {e}")
        sys.exit(1)
    
    # surrogate 모델 학습 (예측값을 근사할 모델)
    surrogate = train_surrogate_model(X, y)
    
    # surrogate 모델을 이용하여 SHAP 값 계산 및 시각화
    compute_shap_from_surrogate(surrogate, X)

if __name__ == "__main__":
    main()