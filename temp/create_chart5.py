import tensorflow as tf
import shap
import numpy as np

# 1. 먼저, 사용자 정의 gradient를 등록합니다.
@tf.RegisterGradient("shap_AddV2")
def _shap_add_v2_grad(op, grad):
    # AddV2의 gradient는 각 입력에 대해 단순히 grad를 전달하는 것으로 충분할 수 있습니다.
    return grad, grad

# 2. 기존 그래프에서 "AddV2" 연산에 대해 "shap_AddV2" gradient를 사용하도록 오버라이드합니다.
# (주의: 아래 코드는 TF 1.x 스타일의 graph 모드에서 작동합니다. TF2에서는 eager 모드가 기본이므로
#  이 방법을 적용하려면 tf.compat.v1.disable_eager_execution() 등을 사용하거나, 커스텀 오버라이드 컨텍스트를 사용해야 합니다.)
with tf.compat.v1.get_default_graph().gradient_override_map({"AddV2": "shap_AddV2"}):
    # 예시: 모델과 백그라운드 데이터가 준비되었다고 가정
    # model: tf.keras 모델, background: numpy array (예: X의 일부)
    explainer = shap.DeepExplainer(model, background)
    shap_values = explainer.shap_values(X)