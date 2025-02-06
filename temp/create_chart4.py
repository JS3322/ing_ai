import sys
import pandas as pd
import numpy as np
import tensorflow as tf
import shap
import matplotlib.pyplot as plt

def main():
    """
    사용법:
      python script.py <모델_경로> <CSV_경로> <피처컬럼목록> <결과컬럼목록>
      
    예)
      python script.py model.h5 prediction.csv x,y,z sum,multiple,sum-multiple
      
    CSV 파일은 모델 입력에 해당하는 피처 컬럼들과 실제 또는 예측된 결과 컬럼들이 모두 포함되어 있어야 합니다.
    """
    if len(sys.argv) < 5:
        print("Usage: python script.py <모델_경로> <CSV_경로> <피처컬럼목록> <결과컬럼목록>")
        sys.exit(1)
    
    # 인자 처리
    model_path = sys.argv[1]
    csv_path = sys.argv[2]
    feature_columns = sys.argv[3].split(",")   # 예: "x,y,z" → ['x', 'y', 'z']
    result_columns = sys.argv[4].split(",")      # 예: "sum,multiple,sum-multiple"
    
    # 1. TensorFlow 모델 로드
    model = tf.keras.models.load_model(model_path)
    print(f"TensorFlow 모델을 '{model_path}'에서 로드했습니다.")
    
    # 2. CSV 파일 읽기 및 피처 데이터 추출
    df = pd.read_csv(csv_path)
    X = df[feature_columns].values.astype(np.float32)
    print(f"CSV 파일 '{csv_path}'에서 피처 컬럼 {feature_columns}를 추출했습니다. (총 샘플 수: {len(X)})")
    
    # 3. DeepExplainer에 사용할 백그라운드 데이터 (전체 데이터 중 일부)
    background = X[:100] if len(X) >= 100 else X
    
    # 4. DeepExplainer 생성 및 SHAP 값 계산
    explainer = shap.DeepExplainer(model, background)
    shap_values = explainer.shap_values(X)
    
    # 5. SHAP Summary Plot 생성 및 저장
    # 모델이 다중 출력을 가지면 shap_values는 리스트로 반환됩니다.
    if isinstance(shap_values, list):
        for i, sv in enumerate(shap_values):
            col_name = result_columns[i] if i < len(result_columns) else f"output_{i}"
            shap.summary_plot(sv, X, feature_names=feature_columns, show=False)
            plt.title(f"SHAP Summary for {col_name}")
            filename = f"shap_summary_{col_name}.png"
            plt.savefig(filename)
            plt.close()
            print(f"SHAP summary plot saved as '{filename}'.")
    else:
        # 단일 출력인 경우
        shap.summary_plot(shap_values, X, feature_names=feature_columns, show=False)
        title = result_columns[0] if result_columns else "output"
        plt.title(f"SHAP Summary for {title}")
        plt.savefig("shap_summary.png")
        plt.close()
        print("SHAP summary plot saved as 'shap_summary.png'.")

if __name__ == "__main__":
    main()