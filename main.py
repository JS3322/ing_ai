from src.doe.interface.make_data_interface import HBMSyntheticDataGenerator
from src.ml.interface.train_interface import HBMBandwidthModel, check_gpu

# if __name__ == "__main__":
#     data_generator = HBMSyntheticDataGenerator(n_samples=1000, output_dir='reference')
#     data_generator.generate_data()
#     data_generator.split_and_shuffle(train_ratio=0.8)
#     data_generator.save_data()

if __name__ == "__main__":
    check_gpu()  # GPU 사용 가능 여부 확인
    hbm_model = HBMBandwidthModel()
    hbm_model.load_data()
    hbm_model.create_model()
    hbm_model.train_model(epochs=100, batch_size=32)
    hbm_model.evaluate_model()
    hbm_model.save_model()  # 기본적으로 saved_model/model.h5에 저장