from src.doe.application.make_data_interface import HBMSyntheticDataGenerator

if __name__ == "__main__":
    data_generator = HBMSyntheticDataGenerator(n_samples=1000, output_dir='reference')
    data_generator.generate_data()
    data_generator.split_and_shuffle(train_ratio=0.8)
    data_generator.save_data()