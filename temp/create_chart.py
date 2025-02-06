import sys
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import pandas as pd

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

def predict_function(model, X):
    """
    모델의 예측 함수.
    모델 종류에 따라 적절히 예측값을 반환합니다.
    """
    # 1. PyTorch 모델인 경우
    try:
        import torch
        if isinstance(model, torch.nn.Module):
            # numpy array를 torch tensor로 변환
            X_tensor = torch.tensor(X, dtype=torch.float32)
            with torch.no_grad():
                predictions = model(X_tensor)
            # tensor이면 numpy array로 변환 후 반환
            if isinstance(predictions, torch.Tensor):
                return predictions.cpu().numpy()
            else:
                return predictions
    except ImportError:
        pass  # torch가 설치되어 있지 않은 경우 넘어감

    # 2. Keras/TensorFlow 또는 scikit-learn 기반 모델인 경우
    if hasattr(model, "predict"):
        try:
            # 분류 문제의 경우 predict_proba가 있을 수 있음
            if hasattr(model, "predict_proba"):
                return model.predict_proba(X)
            else:
                return model.predict(X)
        except Exception as e:
            print("예측 실행 중 오류 발생:", e)
            sys.exit(1)
    else:
        print("모델에 predict 메서드가 없습니다.")
        sys.exit(1)

def compute_shap_feature_importance(model_path, X):
    """
    모델 파일 경로와 피처로 구성된 입력 데이터(X)를 받아
    SHAP KernelExplainer로 피처 영향도를 계산하고 summary plot을 저장합니다.
    """
    # 모델 로드
    model = load_model(model_path)
    
    # 백그라운드 데이터: 입력 데이터의 처음 100개 샘플 사용
    background = X[:min(100, X.shape[0])]
    
    # 예측 함수를 람다 함수로 정의
    pred_func = lambda x: predict_function(model, x)
    
    # SHAP KernelExplainer 생성 (모델 독립적 방법)
    explainer = shap.KernelExplainer(pred_func, background)
    
    # 전체 데이터를 대상으로 SHAP 값 계산
    shap_values = explainer.shap_values(X)
    
    # SHAP summary plot 생성 및 저장
    shap.summary_plot(shap_values, X, show=False)
    plt.savefig("shap_summary.png")
    plt.close()
    print("SHAP summary plot이 'shap_summary.png'로 저장되었습니다.")

def main():
    """
    사용법:
      python script.py <모델_파일경로> <csv_파일경로> <피처컬럼목록>
    
    <피처컬럼목록>은 쉼표로 구분된 컬럼 이름 문자열로 전달합니다.
    예) python script.py model.h5 data.csv col1,col2,col3
    """
    if len(sys.argv) < 4:
        print("사용법: python script.py <모델_파일경로> <csv_파일경로> <피처컬럼목록>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    csv_path = sys.argv[2]
    # 쉼표로 구분된 피처 컬럼 목록을 배열로 변환
    feature_columns = sys.argv[3].split(",")
    
    # CSV 파일 로드
    try:
        df = pd.read_csv(csv_path)
        print(f"CSV 파일 '{csv_path}' 로드 성공. 총 샘플 수: {len(df)}")
    except Exception as e:
        print(f"CSV 파일 로드 실패: {e}")
        sys.exit(1)
    
    # CSV 데이터에서 지정한 피처 컬럼만 선택
    try:
        X = df[feature_columns].values
    except KeyError as e:
        print(f"CSV 파일에 지정된 피처 컬럼이 존재하지 않습니다: {e}")
        sys.exit(1)
    
    compute_shap_feature_importance(model_path, X)

if __name__ == "__main__":
    main()