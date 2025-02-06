import sys
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import tensorflow as tf

# torch가 설치되어 있을 경우만 임포트
try:
    import torch
except ImportError:
    torch = None

def load_model(model_path):
    """
    모델 경로로부터 모델을 로드합니다.
    joblib, tf.keras, torch 순으로 로드를 시도합니다.
    """
    # 1. scikit-learn 등 joblib로 저장된 모델
    try:
        model = joblib.load(model_path)
        print("joblib를 사용해 모델 로드 성공.")
        return model
    except Exception as e:
        print("joblib 모델 로드 실패:", e)
    
    # 2. tf.keras (또는 TensorFlow) 모델 (.h5 등)
    try:
        model = tf.keras.models.load_model(model_path)
        print("Keras/TensorFlow 모델 로드 성공.")
        return model
    except Exception as e:
        print("Keras/TensorFlow 모델 로드 실패:", e)
    
    # 3. PyTorch 모델 (torch.nn.Module)
    if torch is not None:
        try:
            model = torch.load(model_path, map_location=torch.device("cpu"))
            model.eval()  # 평가 모드로 전환
            print("PyTorch 모델 로드 성공.")
            return model
        except Exception as e:
            print("PyTorch 모델 로드 실패:", e)
    
    sys.exit("모델 로드에 실패하였습니다.")

def predict_function(model, X):
    """
    모델의 predict 함수를 호출하여 예측 결과를 반환합니다.
    모델의 종류에 따라 적절하게 처리합니다.
    """
    # PyTorch 모델인 경우
    if torch is not None and isinstance(model, torch.nn.Module):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            preds = model(X_tensor)
        if isinstance(preds, torch.Tensor):
            return preds.cpu().numpy()
        return preds

    # tf.keras 또는 scikit-learn 모델인 경우
    if hasattr(model, "predict"):
        try:
            return model.predict(X)
        except Exception as e:
            print("predict 함수 실행 중 오류:", e)
            sys.exit(1)
    else:
        sys.exit("모델에 predict 메서드가 없습니다.")

def main():
    """
    사용법:
      python script.py <모델_경로> <예측결과_csv_경로> <피처컬럼목록> <결과컬럼목록>
    
    예)
      python script.py model.h5 prediction.csv x,y,z sum,multiple,sum-multiple

    - 피처컬럼목록과 결과컬럼목록은 쉼표로 구분된 문자열을 배열로 변환합니다.
    - 예측 결과 CSV에는 모델 입력에 해당하는 피처 컬럼들과 모델의 출력(예측 혹은 실제 y) 컬럼들이 포함되어야 합니다.
    """
    if len(sys.argv) < 5:
        print("사용법: python script.py <모델_경로> <예측결과_csv_경로> <피처컬럼목록> <결과컬럼목록>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    csv_path = sys.argv[2]
    feature_columns = sys.argv[3].split(",")      # 예: "x,y,z" → ['x', 'y', 'z']
    result_columns = sys.argv[4].split(",")         # 예: "sum,multiple,sum-multiple"

    # 1. 모델 로드
    model = load_model(model_path)
    
    # 2. CSV 파일 읽기
    try:
        df = pd.read_csv(csv_path)
        print(f"CSV 파일 '{csv_path}' 로드 성공. 총 샘플 수: {len(df)}")
    except Exception as e:
        print("CSV 파일 로드 실패:", e)
        sys.exit(1)
    
    # 3. CSV에서 피처(X)와 결과(y) 컬럼 추출
    try:
        X = df[feature_columns].values
    except KeyError as e:
        print("지정한 피처 컬럼이 CSV에 없습니다:", e)
        sys.exit(1)
    
    try:
        Y = df[result_columns].values  # Y 값은 SHAP 계산 시 직접 사용하지는 않지만, 확인용으로 추출합니다.
    except KeyError as e:
        print("지정한 결과 컬럼이 CSV에 없습니다:", e)
        sys.exit(1)
    
    # 4. 모델의 predict 함수를 호출하는 람다 함수 정의
    model_predict = lambda data: predict_function(model, data)
    
    # 5. SHAP Explainer 선택
    # 트리 기반 모델 (feature_importances_ 속성이 있는 경우)라면 TreeExplainer 사용,
    # 아니라면 모델-agnostic 방법인 KernelExplainer 사용.
    if hasattr(model, "feature_importances_"):
        print("트리 기반 모델로 판단되어 TreeExplainer 사용.")
        explainer = shap.TreeExplainer(model)
    else:
        print("KernelExplainer 사용 (모델-agnostic).")
        # KernelExplainer는 백그라운드 데이터가 필요하므로 X의 일부 샘플 사용
        background = X[np.random.choice(X.shape[0], min(100, X.shape[0]), replace=False)]
        explainer = shap.KernelExplainer(model_predict, background)
    
    # 6. SHAP 값 계산
    try:
        shap_values = explainer.shap_values(X)
    except Exception as e:
        print("SHAP 값 계산 중 오류:", e)
        sys.exit(1)
    
    # 7. SHAP Summary Plot 생성 및 저장
    # 모델의 출력이 다중 값(예: 다중 출력 모델)인 경우, shap_values는 리스트로 반환됩니다.
    if isinstance(shap_values, list):
        # 결과 컬럼 목록의 각 값에 대해 별도 plot 생성
        for i, col in enumerate(result_columns):
            shap.summary_plot(shap_values[i], X, feature_names=feature_columns, show=False)
            plt.title(f"SHAP Summary for {col}")
            filename = f"shap_summary_{col}.png"
            plt.savefig(filename)
            plt.close()
            print(f"'{col}'에 대한 SHAP Summary Plot을 '{filename}'로 저장하였습니다.")
    else:
        # 단일 출력인 경우
        shap.summary_plot(shap_values, X, feature_names=feature_columns, show=False)
        title = result_columns[0] if result_columns else "output"
        plt.title(f"SHAP Summary for {title}")
        filename = "shap_summary.png"
        plt.savefig(filename)
        plt.close()
        print(f"SHAP Summary Plot을 '{filename}'로 저장하였습니다.")

if __name__ == "__main__":
    main()