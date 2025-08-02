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

#### 프로젝트 실행환경

##### 의존성 관리
- **Python 버전**: 3.10 이상 요구 (pyproject.toml 기준)
- **패키지 매니저**: `uv` (권장) 또는 `pip`
- **가상환경**: `.venv` 디렉토리 사용

##### uv를 이용한 환경 설정 (권장)
```bash
# uv 설치 (없는 경우)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 프로젝트 의존성 설치
uv sync

# 가상환경 활성화
source .venv/bin/activate  # Linux/Mac
# 또는
.venv\Scripts\activate     # Windows
```

##### pip를 이용한 환경 설정 (대안)
```bash
# 가상환경 생성
python -m venv .venv

# 가상환경 활성화
source .venv/bin/activate  # Linux/Mac
# 또는
.venv\Scripts\activate     # Windows

# 의존성 설치
pip install -r requirements.txt
```

#### main.py 사용법

메인 스크립트는 JSON 설정 파일을 입력으로 받아 step에 따라 다른 작업을 수행합니다.

##### 기본 사용법
```bash
python main.py <json_config_path>
```

##### 지원하는 Step들

###### 1. 데이터 생성 (makedata)
```bash
python main.py _source/example/request_makedata.json
```
설정 파일 예시 (`_source/example/request_makedata.json`):
```json
{
    "step": "makedata",
    "n_samples": 1000,
    "output_dir": "reference/data",
    "train_ratio": 0.8
}
```

###### 2. 모델 학습 (train)
```bash
python main.py _source/example/request_train.json
```
설정 파일 예시 (`_source/example/request_train.json`):
```json
{
    "step": "train",
    "data_path": "reference/data",
    "model_path": "reference/models",
    "epochs": 100,
    "batch_size": 32
}
```

###### 3. 추론 (inference)
```bash
python main.py _source/example/request_inference.json
```
설정 파일 예시 (`_source/example/request_inference.json`):
```json
{
    "step": "inference",
    "data_path": "reference/data",
    "model_path": "reference/models",
    "epochs": 100,
    "batch_size": 32,
    "question": "윈도우와 macOS를 비교하면 macOS가 부드러운 ux를 제공합니다. 왜 이렇게 차이가 나는지 기술적으로 설명하시오"
}
```

##### 환경 변수 설정

프로젝트는 `.env` 파일을 통해 환경 변수를 관리합니다. 다음과 같은 환경 변수를 설정할 수 있습니다:

```bash
# .env 파일 예시
DEFAULT_N_SAMPLES=1000
DEFAULT_TRAIN_RATIO=0.8
DEFAULT_EPOCHS=100
DEFAULT_BATCH_SIZE=32
DEFAULT_MODEL_CACHE_DIR=reference/models
HUGGINGFACE_TOKEN=your_hf_token_here
```

##### 프로젝트 구조
```
project_ai_model/
├── src/                    # 소스 코드 모듈
├── _source/               # 데이터 및 설정 파일
│   ├── data/             # 학습/테스트 데이터
│   └── example/          # JSON 설정 파일 예시
├── reference/             # 모델 및 참조 데이터
├── main.py               # 메인 실행 스크립트
├── pyproject.toml        # uv 프로젝트 설정
├── requirements.txt      # pip 의존성 파일
└── README.md            # 프로젝트 문서
```
---

#### 기타 내용
```
(정의)Deck : 시뮬레이션에서 필요한 공정 조건, 파라미터, 시나리오 등을 구체적으로 기술한 설정 파일(스크립트)을 의미 (Deck은 ‘무엇을, 어떻게 시뮬레이션할 것인가’를 사람이 이해하기 쉬운 형식으로 작성한 **시뮬레이션 사용설명서(스크립트)) (웨이퍼 정보, 산화공정 시간·온도, 포토리소그래피 마스크 종류, 식각 방식, 도핑 농도, 열처리 온도·시간 등이 단계별로 정리)
```
# (예시) 시뮬레이션 Deck 
1) WAFER: TYPE=Silicon, THICKNESS=775um 
2) OXIDATION: TEMP=1000C, TIME=30min, MODEL=Deal-Grove 
3) PHOTOLITHOGRAPHY: MASK=Mask1, RESIST_THICKNESS=1um 
4) ETCHING: TYPE=Plasma, GAS=CF4, TIME=60s 
5) DOPING: SPECIES=Boron, CONCENTRATION=1e15, ENERGY=50keV 
6) ANNEALING: TEMP=900C, TIME=20min 
7) ...
```

(정의)Backbone : 이 Deck을 해석·실행하여 실제 물리 모델링과 계산(시뮬레이션)을 담당하는 기본 엔진(코어)을 뜻함 (Deck에서 정의된 공정 단계와 물리·화학적 파라미터를 실제 계산해주는 핵심 프로그램) (반도체 물성과 연관된 복잡한 미분방정식(확산 방정식, 포아송 방정식 등)을 풀어서, 농도 분포, 두께 변화, 전기적 특성 변화 등을 시간에 따라 시뮬레이션) (Backbone은 위 Deck을 순차적으로 읽고, 각 단계에 맞는 물리 모델(Deal-Grove 산화 모델, 플라즈마 식각 모델, 확산 모델 등)을 적용해 계산)

(정의) 반도체 공정 시뮬레이션의 TCAD 란 무엇입니까?
Technology Computer-Aided Design 으로 반도체 소자의 물리적·전기적 특성을 시뮬레이션하고 해석하기 위해 사용되는 소프트웨어 툴이나 시뮬레이션 환경을 일컫으며, 반도체 소자의 공정(식각, 확산, 이온주입, 열처리 등)부터 소자 구조(트랜지스터, 다이오드 등) 내부의 물리 현상(전류, 전압, 캐리어 농도, 전계 분포 등)을 정확한 물리 모델로 계산.

(정의) 반도체 공정의 Etch, CVD, RTP는 무엇을 뜻합니까?
Etch (식각, 에칭) : 박막(thin film)이나 웨이퍼 표면의 일부 물질을 제거(식각)하는. 공정
CVD (Chemical Vapor Deposition, 화학 기상 증착) : 기상(가스 상태)의 전구체(Precursor) 물질을 웨이퍼 표면에 화학반응을 일으켜 박막을 형성하는 공정으로 트랜지스터 게이트 산화막(예: SiO₂), 도핑된 폴리실리콘(Poly-Si), 질화막(SiN), 금속 박막(예: W, TiN) 등 다양한 박막을 균일하게 증착.
RTP (Rapid Thermal Processing, 급속 열처리) : 웨이퍼를 짧은 시간에 급속 가열·냉각하여 특정 물리·화학적 변화를 유도하는 공정
(MOSFET(금속 산화막 반도체 트랜지스터) 공정 흐름 예시)
웨이퍼 준비 (Wafer Preparation) -> 산화(Oxidation) & 박막 증착(Deposit)으로 CVD(화학 기상 증착) -> 포토리소그래피(Photolithography) 감광액 도포 -> Etch(식각) -> 이온주입(Ion Implantation) & 열처리(RTP) -> 메탈 증착(Metallization) & 추가 식각 -> 패키징(Packaging) 전 공정 마무리
1. 웨이퍼로 도화지 생성 2. 회의 그리기(포토리소그래피) 3. 깍아내기 (식각, Etching) 4. 반도체 성질 넣기 (이온 주입, Doping) 에서 2번에서 4번 반복하여 layer생성

(개선) (prediction 재학습 및 stwinner 제공 서비스를 재학습 모델을 자동화 예제) 공정 레시피(Recipe) 조건을 시점마다 조정, 장비 가동 순서/조건에 대해 자세히 설명하고, Reinforcement Learning (RL)의 예제를 작성하시오.
공정 레시피(Recipe)란, 예컨대 **에칭(Etch)**이나 CVD(증착), 확산(열처리) 등의 공정을 진행할 때 온도, 압력, 가스 플로우, 시간, 전력 등 각종 파라미터를 어떻게 설정하고, 어느 시점에 어떤 장비를 가동할지 등을 구체적으로 기술한 공정 조건표를 의미
:: 특정모델을 올려놓고, prediction을 통해 in/output 데이터가 존재하면 강화학습을 통해 모델 재생성 후 serving을 자동으로 제공하는 mlops

(기존flow확인:Transfer Learning(전이학습))여러 종류의 시뮬레이터(고정밀 but 느린 모델 vs. 저정밀 but 빠른 모델)가 존재하거나, 하드웨어/메타 모델별로 fidelity(정확도)가 다를 때에 대해 자세히 설명하고, Transfer Learning 기반 DNN (저해상도/저정밀 데이터로 pre-training → 고정밀 데이터로 fine-tuning)의 예제를 작성하시오.
Multi-Fidelity / Transfer Learning으로 저정밀 시뮬레이션(Low-Fidelity) 모델(예비 학습)을 갖고 하드웨어 데이터 기반 High-Fidelity을 Fine-tuning 하여 최종 모델을 고정밀 예측 능력으로 보정

(기존flow확인:Transfer Learning(전이학습))저정밀 deck의 정보(통계적/경험적 정보)와 고정밀 deck 정보(산화공정에서 성장 기구, 계면 반응, 응력(Stress) 등까지 고려..)을 분류하여, backbone 진행

(기존flow확인:Transfer Learning(전이학습))또는 deck 정보를 저정밀 backbone(2D 전용 또는 간단한 PDE 해석 모듈만 탑재된 시뮬레이터)과 고정밀 backbone(풀스케일 TCAD 엔진(예: 편미분방정식 솔버, 복합 물성 DB)) 진행

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

*각각 모델을 연동(input/output 잇고, 모델을 연결하는 추천/연결시스템)하는 ops구성
(확장:모델) 공정 시작 전에 웨이퍼 특성을 측정하거나, 과거 생산 이력(메타데이터)으로부터 표면 상태를 예측으로 표면 상태가 불량인 웨이퍼는 사전에 걸러내거나, 다음 공정 파라미터를 보정(온도, 시간 등)하여 결과 편차를 줄이는 모델 알고리즘 적용
(확장:모델) 산화 온도·시간·압력 등 공정 파라미터가 주어지면 최종 산화막 두께를 예측하는 ML 모델로 공정 설계 단계에서 원하는 산화막 두께(예: 10nm)를 얻기 위한 파라미터 조건을 빠르게 찾음
(확장:모델) 마스크(Mask)에 그려진 회로 패턴이 실제 웨이퍼의 포토레지스트(PR)에 어떻게 형성되는지 예측으로 다양한 노광 조건(파장, NA, 투과율 등)에 따른 패턴 형성 결과(CD, Line Profile)를 예측
(확장:모델) 노광 후 현상 공정에서 포토레지스트가 어디까지 제거되며, 패턴 테두리가 어떻게 형성되는지 예측으로 “포토 공정 파라미터(노광 시간, 현상 시간 등)에 따른 실제 라인폭 편차”를 빠르게 추정 & 공정 조건 최적화
(확장:모델) 공정 변동(Defocus, Dose error)으로 인해 목표 라인폭과 실제 라인폭이 다르게 나오는 문제를 포토 공정 시뮬레이션 데이터(광학, 레지스트 현상 결과)를 학습해 공정 편차에 따른 CD 변화를 예측
(확장:모델) 공정 파라미터(가스 조성, 압력, 전력, 시간) → 식각 깊이, 프로파일 모양을 회귀 or CNN으로 예측하여 목표 패턴의 선폭, 측벽 각도 등을 맞추기 위해 플라즈마 식각 공정 조건을 최적화
(확장:모델) Reinforcement Learning 등을 활용해 “균일도를 최대화”하는 조건(전력, 압력, 가스 유량 등) 탐색하기 위해 식각 후 패턴 편차를 줄여서 불량률(Yield)을 높임
(확장:모델) 이온 주입단계에서 시뮬레이션/실험 data(에너지·각도·선량 등) → 분포 곡선 매개변수(피크 농도, 분포 폭 등)를 예측하는 Regression NN으로 원하는 pn 접합 깊이를 달성하기 위한 도핑 조건을 빠르게 결정
(확장:모델) 도핑 프로파일 전후 변화를 학습한 NN 또는 GPR로 열처리 조건(온도, 시간 등)에 따른 최종 농도분포 예측하여 실제 소자 특성(Threshold Voltage 등)에 맞게 주입+열처리 공정을 최적화
(확장:모델) 공정 반복 시 다층(Layer) 구조 형성 시뮬레이터에서 레이어가 많아질수록 누적 오차(두께, 농도, 식각 각도)가 발생 → 최종 소자 특성 편차 증가하므로 Multi-step Regression 각 공정 단계별 파라미터 + 이전 단계 결과 → 다음 단계 결과를 예측하여 여러 레이어 쌓인 최종 구조(두께, 도핑 분포, 공정 결함 등)를 빠르게 예측해 공정 Recipe를 조정
(확장:모델) 패키징 과정에서 열이나 외부 충격이 가해질 때 칩 크랙(crack), 델라미네이션(박리) 발생하므로 응력·온도·소재 특성 → 파손 확률 예측 분류(Classification) 모델, 또는 lifetime 예측 회귀모델로 패키지 재질/두께 설계 최적화, 신뢰성(Reliability) 향상
```