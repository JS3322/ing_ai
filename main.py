import json
import argparse
import os
from src.doe.interface.make_data_interface import HBMSyntheticDataGenerator
from src.ml.interface.train_interface import HBMBandwidthModel, check_gpu

def main(json_path):
    """
    JSON 파일을 읽어서 step 키 값에 따라 작업을 수행하는 함수
    
    Args:
        json_path (str): JSON 설정 파일의 경로
    """
    # JSON 파일 존재 여부 확인
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON 파일을 찾을 수 없습니다: {json_path}")
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON 파일 형식이 잘못되었습니다: {e}")
    
    # 필수 키 확인
    if 'step' not in config:
        raise KeyError("JSON 파일에 'step' 키가 없습니다.")
    
    step = config.get('step')
    
    if step == 'makedata':
        # 데이터 생성 단계
        data_generator = HBMSyntheticDataGenerator(
            n_samples=config.get('n_samples', 1000),
            output_dir=config.get('output_dir', 'reference')
        )
        data_generator.generate_data()
        data_generator.split_and_shuffle(train_ratio=config.get('train_ratio', 0.8))
        data_generator.save_data()
    
    elif step == 'train':
        # 모델 학습 단계
        check_gpu()
        hbm_model = HBMBandwidthModel()
        hbm_model.load_data()
        hbm_model.create_model()
        hbm_model.train_model(
            epochs=config.get('epochs', 100),
            batch_size=config.get('batch_size', 32)
        )
        hbm_model.evaluate_model()
        hbm_model.save_model()
    
    else:
        raise ValueError(f"지원하지 않는 step입니다: {step}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='HBM 모델 학습 및 데이터 생성 스크립트')
    parser.add_argument('json_path', type=str, 
                      help='설정 JSON 파일의 경로 (예: _source/example/request_makedata.json)')
    
    try:
        args = parser.parse_args()
        main(args.json_path)
    except SystemExit as e:
        print("\n사용법: python main.py <json_path>")
        print("예시: python main.py _source/example/request_makedata.json")
        exit(1)
    except Exception as e:
        print(f"오류 발생: {str(e)}")
        exit(1)