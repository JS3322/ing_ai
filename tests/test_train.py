import os
import json
import pytest
import shutil
import numpy as np
import pandas as pd
from src.ml.interface.train_interface import HBMBandwidthModel
from main import main

class TestTrainStep:
    @pytest.fixture
    def setup_test_data(self):
        """테스트용 데이터 생성"""
        # 테스트 데이터 디렉토리 생성
        test_data_path = "test_data"
        os.makedirs(test_data_path, exist_ok=True)

        # 테스트용 데이터 생성
        n_samples = 100
        np.random.seed(42)
        
        # 특성 데이터 생성
        data = {
            'layer_distance': np.random.uniform(1, 10, n_samples),
            'interconnect_width': np.random.uniform(0.1, 1, n_samples),
            'material_resistivity': np.random.uniform(1e-8, 1e-6, n_samples),
            'dielectric_constant': np.random.uniform(1, 5, n_samples),
            'temperature': np.random.uniform(20, 100, n_samples)
        }
        
        # 목표 변수 (bandwidth) 생성 - 간단한 계산식 사용
        data['bandwidth'] = (data['interconnect_width'] * 1e3) / (data['layer_distance'] * data['material_resistivity'])
        data['bandwidth'] += np.random.normal(0, 0.1, n_samples)  # 노이즈 추가
        
        # 훈련/테스트 데이터 분할
        train_size = int(n_samples * 0.8)
        df = pd.DataFrame(data)
        train_df = df[:train_size]
        test_df = df[train_size:]
        
        # CSV 파일로 저장
        train_df.to_csv(os.path.join(test_data_path, "train.csv"), index=False)
        test_df.to_csv(os.path.join(test_data_path, "test.csv"), index=False)
        
        yield test_data_path
        
        # 테스트 후 정리
        shutil.rmtree(test_data_path)

    @pytest.fixture
    def setup_test_config(self, setup_test_data):
        """테스트용 설정 파일 생성"""
        config = {
            "step": "train",
            "data_path": setup_test_data,
            "model_path": "test_models",
            "epochs": 2,
            "batch_size": 16
        }
        
        config_path = "test_config.json"
        with open(config_path, "w") as f:
            json.dump(config, f)
            
        yield config_path
        
        # 테스트 후 정리
        os.remove(config_path)
        if os.path.exists("test_models"):
            shutil.rmtree("test_models")

    def test_train_model_creation(self, setup_test_data):
        """모델 생성 테스트"""
        model = HBMBandwidthModel(data_path=setup_test_data)
        model.load_data()
        model.create_model()
        
        assert model.model is not None
        assert len(model.model.layers) == 5  # 입력층 + 3개의 은닉층 + 출력층

    def test_data_loading(self, setup_test_data):
        """데이터 로딩 및 전처리 테스트"""
        model = HBMBandwidthModel(data_path=setup_test_data)
        model.load_data()
        
        assert model.X_train is not None
        assert model.y_train is not None
        assert model.X_test is not None
        assert model.y_test is not None
        
        # 정규화 검증
        assert np.all(model.X_train >= 0) and np.all(model.X_train <= 1)
        assert np.all(model.X_test >= 0) and np.all(model.X_test <= 1)

    def test_model_training(self, setup_test_data):
        """모델 학습 테스트"""
        model = HBMBandwidthModel(data_path=setup_test_data)
        model.load_data()
        model.create_model()
        history = model.train_model(epochs=2, batch_size=16)
        
        assert history is not None
        assert len(history.history['loss']) == 2  # 2 에포크

    def test_model_saving(self, setup_test_data):
        """모델 저장 테스트"""
        model_path = "test_models"
        model = HBMBandwidthModel(data_path=setup_test_data, model_path=model_path)
        model.load_data()
        model.create_model()
        model.train_model(epochs=2, batch_size=16)
        model.save_model()
        
        assert os.path.exists(os.path.join(model_path, "model.h5"))
        
        # 테스트 후 정리
        shutil.rmtree(model_path)

    def test_main_train_workflow(self, setup_test_config):
        """전체 학습 워크플로우 테스트"""
        main(setup_test_config)
        
        # 설정 파일 읽기
        with open(setup_test_config, 'r') as f:
            config = json.load(f)
        
        # 모델 파일이 저장되었는지 확인
        model_path = config.get('model_path', config.get('data_path'))
        assert os.path.exists(os.path.join(model_path, "model.h5"))

    def test_invalid_data_path(self):
        """잘못된 데이터 경로에 대한 예외 처리 테스트"""
        with pytest.raises(FileNotFoundError):
            model = HBMBandwidthModel(data_path="nonexistent_path")
            model.load_data() 