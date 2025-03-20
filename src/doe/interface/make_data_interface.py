import os
import torch
import pandas as pd


class HBMSyntheticDataGenerator:
    def __init__(self, n_samples=1000, output_dir='reference/data'):
        # 재현성을 위한 시드 설정
        torch.manual_seed(42)

        self.n_samples = n_samples
        self.output_dir = output_dir

        # 생성될 DataFrame을 저장할 변수 초기화
        self.df = None
        self.train_df = None
        self.test_df = None

    def generate_data(self):
        """
        PyTorch 연산을 사용해 문헌 기반 파라미터 범위에 따른 합성 데이터를 생성합니다.
        """
        # 층 간 거리: 5 ~ 20 µm
        layer_distance = torch.rand(self.n_samples) * (20 - 5) + 5
        # 인터커넥트 폭: 0.5 ~ 5 µm
        interconnect_width = torch.rand(self.n_samples) * (5 - 0.5) + 0.5
        # 재료 저항: 1 ~ 5 Ω·cm (효과적인 값)
        material_resistivity = torch.rand(self.n_samples) * (5 - 1) + 1
        # 유전율: 2 ~ 10 (예: SiO2 ~ Low-k)
        dielectric_constant = torch.rand(self.n_samples) * (10 - 2) + 2
        # 온도: 25 ~ 85 °C
        temperature = torch.rand(self.n_samples) * (85 - 25) + 25

        # 실제 환경의 불확실성을 반영할 노이즈 (정규분포)
        noise = torch.randn(self.n_samples) * 0.05

        # 대역폭 계산 수식:
        # bandwidth = (인터커넥트 폭 / 층 간 거리) / (재료 저항 * 유전율) * ((100 - 온도)/100) + 노이즈
        base_bandwidth = (interconnect_width / layer_distance) / (material_resistivity * dielectric_constant)
        bandwidth = base_bandwidth * ((100 - temperature) / 100) + noise

        # 음수 값은 0으로 클리핑
        bandwidth = torch.clamp(bandwidth, min=0)

        # 텐서 데이터를 NumPy 배열로 변환 후, DataFrame 생성
        data = {
            'layer_distance': layer_distance.numpy(),
            'interconnect_width': interconnect_width.numpy(),
            'material_resistivity': material_resistivity.numpy(),
            'dielectric_constant': dielectric_constant.numpy(),
            'temperature': temperature.numpy(),
            'bandwidth': bandwidth.numpy()
        }
        self.df = pd.DataFrame(data)
        print("합성 데이터 생성 완료.")

    def split_and_shuffle(self, train_ratio=0.8, random_state=42):
        """
        생성된 데이터를 섞은 후 지정한 비율로 학습 데이터와 테스트 데이터로 분리합니다.
        """
        if self.df is None:
            raise ValueError("먼저 generate_data()를 호출하여 데이터를 생성하세요.")

        df_shuffled = self.df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        train_size = int(train_ratio * self.n_samples)
        self.train_df = df_shuffled.iloc[:train_size]
        self.test_df = df_shuffled.iloc[train_size:]
        print("데이터 분할 완료: 학습 데이터와 테스트 데이터로 분리되었습니다.")

    def save_data(self):
        """
        학습 데이터와 테스트 데이터를 CSV 파일로 저장합니다.
        """
        if self.train_df is None or self.test_df is None:
            raise ValueError("먼저 split_and_shuffle()를 호출하여 데이터를 분할하세요.")

        # 프로젝트 루트 디렉토리 찾기
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
        
        # output_dir을 프로젝트 루트 기준으로 설정
        self.output_dir = os.path.join(project_root, self.output_dir)
        print(f"데이터 저장 경로: {self.output_dir}")
            
        os.makedirs(self.output_dir, exist_ok=True)
        train_path = os.path.join(self.output_dir, 'train.csv')
        test_path = os.path.join(self.output_dir, 'test.csv')
        self.train_df.to_csv(train_path, index=False)
        self.test_df.to_csv(test_path, index=False)
        print(f"합성 데이터가 '{train_path}'와 '{test_path}'에 저장되었습니다.")
