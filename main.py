from src.data_loader import load_training_data
from src.preprocessing import preprocess_pipeline
from src.pca_model import PCAModel

if __name__ == "__main__":
    

    df = load_training_data()
    
    


    df_processed, scaler = preprocess_pipeline(df)
    
    sensor_cols = [col for col in df_processed.columns if "sensor" in col]
    X = df_processed[sensor_cols].values
    pca_model = PCAModel(n_components=6)
    X_pca = pca_model.fit_transform(X)




    print("PCA Shape:", X_pca.shape)
    print("Explained variance Ratio:")
    print(pca_model.explained_variance())






#PCA ANALYSIS FOR EACH COMPONENTS


#PC1 → 60.17%
#PC2 → 13.99%
#PC3 → 6.44%
#PC4 → 2.76%
#PC5 → 2.45%
#PC6 → 2.32%
#Removed sensors with zero variances :  ['sensor_1', 'sensor_10', 'sensor_18', 'sensor_19']
#PCA Shape: (20631, 6)
#Explained variance Ratio:
# [0.60175982 0.13990271 0.06442881 0.02759547 0.02455224 0.02322108]
# total variance = adding all the variances = 88.13%

