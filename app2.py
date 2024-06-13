# import csv
# import librosa
# import os
# import numpy as np
# import random

# file = open('NonThreat_Voice.csv', 'w', newline='')
# with file:
#     writer = csv.writer(file)
#     writer.writerow(header)

# i = 0 
# for filename in os.listdir(f'{non_threat_file_path}'):

#     audio = f'{non_threat_file_path}/{filename}'

#     y, sr = librosa.load(audio, mono=True, duration=30)
#     chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
#     rmse = librosa.feature.rms(y=y)
#     spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
#     spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
#     rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
#     zcr = librosa.feature.zero_crossing_rate(y)
#     mfcc = librosa.feature.mfcc(y=y, n_mfcc = 13, sr=sr)
#     to_append = f'{filename} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'

#     if i % 10 == 0:
#       # print(filename)
#       print(i)
#     i += 1
#     for e in mfcc:
#         to_append += f' {np.mean(e)}'
#     # print(len(to_append.split()))
#     to_append += f' {audio}'
#     # print(len(to_append.split()))
#     data = to_append.split()
#     # print(len(to_append.split()))
#     # break
#     file = open('NonThreat_Voice.csv', 'a', newline='')
#     with file:
#         writer = csv.writer(file)
#         writer.writerow(data)

# print("Done")


import pandas as pd
from sklearn.preprocessing import StandardScaler

# Assuming 'df' is your DataFrame with numeric columns
df = pd.DataFrame({
    'A': [10, 20, 30, 40, 50],
    'B': [5, 15, 25, 35, 45],
    'C': [100, 200, 300, 400, 500]
})

# Extract the numeric columns you want to normalize
columns_to_normalize = ['A', 'B', 'C']
data_to_normalize = df[columns_to_normalize]

# Use StandardScaler to perform Z-score normalization
scaler = StandardScaler()
normalized_data = pd.DataFrame(scaler.fit_transform(data_to_normalize), columns=columns_to_normalize)

# Replace the original columns with the normalized values
df[columns_to_normalize] = normalized_data

# Print the original and normalized DataFrame
print("Original DataFrame:")
print(df)
print("\nNormalized DataFrame (Z-score):")
print(normalized_data)
