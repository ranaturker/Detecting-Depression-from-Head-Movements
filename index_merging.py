import os
import numpy as np
import pandas as pd

# Histogram verilerinin olduğu klasör
histogram_folder = r"C:\Users\yagmu\PycharmProjects\capstone\histogramlar"

# Histogram dosyalarını oku
histogram_files = [f for f in os.listdir(histogram_folder) if f.endswith(".csv")]

# Hasta numaralarını belirle ve sırala
patient_ids = sorted(set([f.split("_")[0] for f in histogram_files]))  # Hasta numaralarını sıralı şekilde al

# 1'den 288'e kadar her indeks için bir dosya oluştur
for index in range(1, 289):  # 1'den 288'e kadar döngü
    index_name = f"Index{index}"
    combined_histograms = []

    # Her hasta için bu indekse ait dosyayı oku
    for patient_id in patient_ids:
        index_file = f"{patient_id}_{index_name}.allhistograms_columns.csv"
        file_path = os.path.join(histogram_folder, index_file)

        if os.path.exists(file_path):  # Dosya mevcutsa oku
            with open(file_path, "r") as f:
                histogram_data = f.read().strip().split("\n")  # Her bir satırı oku
                histogram_data = list(map(float, histogram_data))  # Float türüne çevir
                combined_histograms.append(histogram_data)
        else:  # Eğer dosya yoksa, o hasta için boş değer ekle
            combined_histograms.append([0.0])  # Eksik veri durumunda 0.0 ekle

    # NumPy dizisine çevir ve kaydet
    x_matrix = np.array(combined_histograms)

    # Sonuçları bir CSV dosyasına yaz (sadece histogram değerleri)
    output_csv = f"{index_name}_combined_matrix.csv"
    pd.DataFrame(x_matrix).to_csv(output_csv, index=False, header=False)

    print(f"{index_name} için matris oluşturuldu ve {output_csv} dosyasına kaydedildi.")