import matplotlib
matplotlib.use("Agg") #non-gui backend for terminal execution
import matplotlib.pyplot as plt
import os

from src.preprocessing import preprocess_pipeline
from src.data_loader import load_training_data
from src.pca_model import PCAModel

os.makedirs("results/figures", exist_ok = True)
df_raw = load_training_data()
#taking one engine
engine_raw = df_raw[df_raw["unit_number"] == 1]

sensors_to_plot = ["sensor_2", "sensor_7", "sensor_11"]

for sensor in sensors_to_plot:
    plt.figure()
    plt.plot(engine_raw["time_in_cycles"], engine_raw[sensor])
    plt.xlabel("Cycle")
    plt.ylabel(sensor)
    plt.title(f"Raw {sensor} vs Cycle (Engine 1)")
    plt.savefig(f"results/figures/raw_{sensor}.png")
    plt.close()

#for the preprocessed data
df_processed,scaler = preprocess_pipeline(df_raw)
engine_processed = df_processed[df_processed["unit_number"] == 1]


#RUL plot
plt.figure()
plt.plot(engine_processed["time_in_cycles"], engine_processed["RUL"])
plt.xlabel("Cycle")
plt.ylabel("RUL")
plt.title("RUL vs Cycle (Engine 1)")
plt.savefig("results/figures/rul_vs_cycles.png")
plt.close()

#scclaed sensor plot before vs after
#taking the sensor 2 
sensor = "sensor_2"
plt.figure()
plt.plot(engine_raw["time_in_cycles"], engine_raw[sensor], label = "raw")
plt.plot(engine_processed["time_in_cycles"], engine_processed[sensor], label = "Scaled")
plt.xlabel("Cycle")
plt.ylabel(sensor)
plt.legend()
plt.title(f"{sensor}: Raw vs Scaled")
plt.savefig("results/figures/raw_vs_scaled_sensor2.png")
plt.close()

#for the pca
sensor_cols = [col for col in df_processed.columns if "sensor" in col]
X = df_processed[sensor_cols].values
pca_model = PCAModel(n_components=6)
pca_model.fit(X)

explained_var = pca_model.explained_variance()

plt.figure()
plt.bar(range(1,7), explained_var)
plt.xlabel("Principal component")
plt.ylabel("Explained variance ratio")
plt.title("PCA Explained variance ratio")
plt.savefig("results/figures/pca_explained_variance.png")
plt.close()

if __name__ == "__main__":
    print("Generating the plots for the preprocessed stage")



