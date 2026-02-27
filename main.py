from src.data_loader import load_training_data

if __name__ == "__main__":
    df = load_training_data()
    print(df.head())
    print(df.shape)
