import sys
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

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
        # map_location를 cpu로 지정하여 GPU 없이도 로드 가능하도록 함
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
            # numpy array를 torch tensor로 변환 (dtype은 상황에 따라 조정)
            X_tensor = torch.tensor(X, dtype=torch.float32)
            with torch.no_grad():
                predictions = model(X_tensor)
            # predictions가 tensor라면 numpy array로 변환하여 반환
            if isinstance(predictions, torch.Tensor):
                return predictions.cpu().numpy()
            else:
                return predictions
    except ImportError:
        pass  # torch가 설치되어 있지 않은 경우 넘어감

    # 2. Keras/TensorFlow 또는 scikit-learn 기반 모델의 경우
    if hasattr(model, "predict"):
        try:
            # 분류 문제인 경우 predict_proba가 존재할 수 있음
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
    모델 파일 경로와 입력 데이터(X)를 받아 SHAP KernelExplainer로 피처 영향도를 계산합니다.
    """
    # 모델 로드
    model = load_model(model_path)

    # 백그라운드 데이터(참조 데이터): 전체 데이터 중 일부 샘플 선택
    background = X[np.random.choice(X.shape[0], size=min(100, X.shape[0]), replace=False)]
    
    # 예측 함수 정의 (모델 종류에 관계없이 동작)
    pred_func = lambda x: predict_function(model, x)

    # SHAP KernelExplainer 생성 (모델 독립적)
    explainer = shap.KernelExplainer(pred_func, background)

    # 계산 부담을 줄이기 위해 전체 데이터 대신 일부 데이터 샘플 선택
    sample_size = min(100, X.shape[0])
    X_sample = X[:sample_size]
    shap_values = explainer.shap_values(X_sample)

    # SHAP summary plot 생성 및 이미지 저장
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.savefig("shap_summary.png")
    plt.close()
    print("SHAP summary plot이 'shap_summary.png'로 저장되었습니다.")

def main():
    if len(sys.argv) < 2:
        print("사용법: python script.py <모델_파일경로>")
        sys.exit(1)

    model_path = sys.argv[1]
    
    # 예시용 임의 데이터 (실제 사용 시 모델이 학습했던 데이터나 유사 데이터를 사용하세요)
    # 예: 20개의 피처를 가진 500개의 샘플 데이터
    X = np.random.rand(500, 20)
    
    compute_shap_feature_importance(model_path, X)

if __name__ == "__main__":
    main()