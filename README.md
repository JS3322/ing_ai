#### 인공지능 모델

#### env
- base python : 3.9

#### test case
- 가상환경에서 다음 명령어 실행
```
uvicorn _serve_app:app --host 0.0.0.0 --port 8000
```
- 병렬처리 성능 테스트(Apache Bench)
```
ab -n 1000 -c 10 -p request.json -T application/json http://localhost:8000/execute
```
- _serve_app_ray.py 호출 명령어
```
uvicorn _serve_app_ray:app --host 0.0.0.0 --port 8000
curl -X POST "http://localhost:8000/execute" \
     -H "Content-Type: application/json" \
     -d '{"data": [1, 2, 3, 4, 5]}'

```

#### command
- preprocess
```
python main.py --data_dir ./_source/data --train_data train.csv --preprocess
```
- scale
```
python main.py --data_dir ./_source/data --train_data train.csv --scale
```
- train
```
python main.py --data_dir ./_source/data --train_data train.csv --train --epochs 100
```
- test
```
python main.py --data_dir ./_source/data --test_data test.csv --test
```
- predict
```
python main.py --data_dir ./_source/data --predict_data predict.csv --predict
```
- optimize
```
python main.py --data_dir ./_source/data --train_data train.csv --test_data test.csv --optimize
```
---

#### 기타 내용
```
(정의) 반도체 공정 시뮬레이션의 TCAD 란 무엇입니까?
Technology Computer-Aided Design 으로 반도체 소자의 물리적·전기적 특성을 시뮬레이션하고 해석하기 위해 사용되는 소프트웨어 툴이나 시뮬레이션 환경을 일컫으며, 반도체 소자의 공정(식각, 확산, 이온주입, 열처리 등)부터 소자 구조(트랜지스터, 다이오드 등) 내부의 물리 현상(전류, 전압, 캐리어 농도, 전계 분포 등)을 정확한 물리 모델로 계산.

(정의) 반도체 공정의 Etch, CVD, RTP는 무엇을 뜻합니까?
Etch (식각, 에칭) : 박막(thin film)이나 웨이퍼 표면의 일부 물질을 제거(식각)하는. 공정
CVD (Chemical Vapor Deposition, 화학 기상 증착) : 기상(가스 상태)의 전구체(Precursor) 물질을 웨이퍼 표면에 화학반응을 일으켜 박막을 형성하는 공정으로 트랜지스터 게이트 산화막(예: SiO₂), 도핑된 폴리실리콘(Poly-Si), 질화막(SiN), 금속 박막(예: W, TiN) 등 다양한 박막을 균일하게 증착.
RTP (Rapid Thermal Processing, 급속 열처리) : 웨이퍼를 짧은 시간에 급속 가열·냉각하여 특정 물리·화학적 변화를 유도하는 공정
(MOSFET(금속 산화막 반도체 트랜지스터) 공정 흐름 예시)
웨이퍼 준비 (Wafer Preparation) -> 산화(Oxidation) & 박막 증착(Deposit)으로 CVD(화학 기상 증착) -> 포토리소그래피(Photolithography) 감광액 도포 -> Etch(식각) -> 이온주입(Ion Implantation) & 열처리(RTP) -> 메탈 증착(Metallization) & 추가 식각 -> 패키징(Packaging) 전 공정 마무리

(개선) (prediction 재학습 및 stwinner 제공 서비스를 재학습 모델을 자동화 예제) 공정 레시피(Recipe) 조건을 시점마다 조정, 장비 가동 순서/조건에 대해 자세히 설명하고, Reinforcement Learning (RL)의 예제를 작성하시오.
공정 레시피(Recipe)란, 예컨대 **에칭(Etch)**이나 CVD(증착), 확산(열처리) 등의 공정을 진행할 때 온도, 압력, 가스 플로우, 시간, 전력 등 각종 파라미터를 어떻게 설정하고, 어느 시점에 어떤 장비를 가동할지 등을 구체적으로 기술한 공정 조건표를 의미
:: 특정모델을 올려놓고, prediction을 통해 in/output 데이터가 존재하면 강화학습을 통해 모델 재생성 후 serving을 자동으로 제공하는 mlops

(기존flow확인)여러 종류의 시뮬레이터(고정밀 but 느린 모델 vs. 저정밀 but 빠른 모델)가 존재하거나, 하드웨어/메타 모델별로 fidelity(정확도)가 다를 때에 대해 자세히 설명하고, Transfer Learning 기반 DNN (저해상도/저정밀 데이터로 pre-training → 고정밀 데이터로 fine-tuning)의 예제를 작성하시오.
Multi-Fidelity / Transfer Learning으로 저정밀 시뮬레이션(Low-Fidelity) 모델(예비 학습)을 갖고 하드웨어 데이터 기반 High-Fidelity을 Fine-tuning 하여 최종 모델을 고정밀 예측 능력으로 보정

(확장)반도체 공정에서는 불확실성(uncertainty) 관리가 중요에 대해 설명하고, 모델 예측에 대한 신뢰도(uncertainty)를 추정하면서, “이 구간에선 데이터가 부족하니 다시 실험/시뮬레이션을 해봐야 한다”는 식의 의사결정을 Bayesian Neural Network(BNN)을 통해 어떻게 진행하는지 예제 작성하고, 공정 파라미터를 바꿨을 때의 예측값 ± 오차 범위를 통해, 보수적/안전한 의사결정을 어떻게 하는지 알려주시오.
오차범위 추정 기능으로 Bayesian Neural Network(BNN)은 가중치(Weight)를 확률분포로 취급하여, 학습 과정에서 그 분포(평균 & 분산 등)를 업데이트하고, 예측 시에는 확률적 추론을 통해 예측값 분포를 얻어내는 방식
모델 예측이 표준편차(또는 분산)으로 표현되었을 때, 불확실성이 큰 영역을 어떻게 판단하고, 추가 실험이나 공정 레시피 재설정 여부 (표준편차가 가 특정 임계값)을 넘으면, “해당 파라미터 영역은 모델 자신이 자신감(Confidence)이 떨어진다”라고 인식.. 등등)

(확장)특정 노드(공정 장비)에서 발생한 오류가 다른 노드로 전파 → 장비 간 상호작용(배선, 네트워크)을 고려해야 할 때 웨이퍼 맵, 소자 어레이, 장비 네트워크 등 “그래프/네트워크” 데이터를 모델링을 어떻게 하는지 Graph Neural Network (GNN) 예제를 작성하시오
반도체 공정 장비 네트워크(혹은 웨이퍼 맵, 소자 어레이 등)를 그래프 형태로 모델링하여, Graph Neural Network(GNN)(예: GCN, GAT 등)을 사용하는 PyTorch Geometric (
노드(Node): 공정 장비(혹은 웨이퍼 상의 소자 위치, 배선의 특정 점 등)
엣지(Edge): 장비 간 연결 관계(배선 경로, 네트워크 링크), 또는 소자 간 인접도
노드 피처: 각 장비(노드)에 대한 센서 값, 상태(온도, 압력, 가동률, 결함 이력 등)
노드 레이블: 예) “정상/오류” 상태(2분류), 혹은 장비 그룹(클러스터 라벨), 혹은 결함 유형 등
GNN 모델: 그래프 구조를 이용해, 인접 노드들로부터 정보를 Message Passing하여 노드 상태를 분류/회귀
)

(개선) 특정 알고리즘 모델 추가 건으로 반도체 물성, 전자 이동, 열전달 등 정교한 물리 방정식을 일부 알고 있고, 데이터도 일부 가지고 있을 때 Physics-Informed Neural Networks(PINNs), Hybrid PDE-Net 예제를 작성하시오.
1차원 PDE를 예시로 들어, **물리 방정식(편미분방정식)**과 **데이터(실험/시뮬레이션 결과)**를 함께 반영하는 알고리즘
```