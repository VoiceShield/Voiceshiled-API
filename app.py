from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
from fastapi.responses import JSONResponse
import numpy as np
import librosa
import pandas as pd
from io import StringIO
import re

from sklearn.preprocessing import StandardScaler

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can specify a list of allowed origins or use "*" for all
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class LoadModel:
    def load_cnn_model(self, cnn_model_path):
        cnn_model = joblib.load(cnn_model_path)
        return cnn_model
    def load_tfidf_vectorizer(self, tfidf_file_path):
        tfidf_vectorizer = joblib.load(tfidf_file_path)
        return tfidf_vectorizer
    def load_audio_model(self, model_path):
        return joblib.load(model_path)
    
    def extract_features(self, file_path):
        with open(file_path, "rb") as audio_file:
            audio, sample_rate = librosa.load(audio_file, res_type='kaiser_fast')
        mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13), axis=1)
        return mfccs

def extract_features(file_path):
    with open(file_path, "rb") as audio_file:
        y, sr = librosa.load(audio_file, mono=True, duration=30)
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        rmse = librosa.feature.rms(y=y)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        mfcc = librosa.feature.mfcc(y=y, n_mfcc = 13, sr=sr)
        to_append = f'{np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'
        
        for e in mfcc:
            to_append += f' {np.mean(e)}'
        
    return to_append


# Define input data model
class InputText(BaseModel):
    text: str
    
@app.post("/api/v1/predict/text")
async def predict_hate_speech(input_text: InputText):
    preprocessed_text = preprocess_input(input_text.text)
    
    model_loader = LoadModel()
    tfidf_vectorizer_path = "tfidf_vectorizer.pkl"
    model_path = "text_model.pkl"
    
    
    cnn_model = model_loader.load_cnn_model(model_path)
    tfidf_vectorizer = model_loader.load_tfidf_vectorizer(tfidf_vectorizer_path)
    
    print("Loaded succusfully")
    

    # Vectorize the text using TF-IDF vectorizer
    vectorized_text = tfidf_vectorizer.transform([preprocessed_text]).toarray()

    probability = cnn_model.predict(vectorized_text)
    
    custom_predictions = (probability> 0.5).astype(int).flatten()
    
    data = {
        "predicted_class": str(custom_predictions[0]),
        "confidence": str(probability[0][0])
    }
    return JSONResponse(content=data)

@app.post("/api/v3/predict/text")
async def predict_threat_speech(input_text: InputText):
    preprocessed_text = clean_text(input_text.text)
    print(preprocessed_text)
    model_loader = LoadModel()
    tfidf_vectorizer_path = "vectorizer_reshaped.pkl"
    model_path = "cnn_model_threat.pkl"
    
    
    cnn_model = model_loader.load_cnn_model(model_path)
    tfidf_vectorizer = model_loader.load_tfidf_vectorizer(tfidf_vectorizer_path)
    
    print("Loaded successfully")
    
    # Vectorize the text using TF-IDF vectorizer
    vectorized_text = tfidf_vectorizer.transform([preprocessed_text]).toarray()
    
    print("Vectorized text shape:", vectorized_text.shape)
    print("Model input shape:", cnn_model.input_shape)

    if vectorized_text.shape[1] != cnn_model.input_shape[-1]:
        return JSONResponse(content={"error": "Input shape mismatch. Please ensure the TF-IDF vectorizer and model are compatible."})

    # Predict using the model
    predictions_prob = cnn_model.predict(vectorized_text)
    predicted_class = np.argmax(predictions_prob, axis=1)[0]

    custom_predictions = (predictions_prob> 0.5).astype(int).flatten()
    
    data = {
        "predicted_class": str(custom_predictions[0]),
        "confidence": str(predictions_prob[0][0])
    }
    return JSONResponse(content=data)


@app.post("/api/v4/predict/text")
async def predict_threat_lstm_speech(input_text: InputText):
    preprocessed_text = clean_text(input_text.text)
    print(preprocessed_text)
    model_loader = LoadModel()
    tfidf_vectorizer_path = "vectorizer_reshaped.pkl"
    model_path = "lstm_model_final.pkl"
    
    
    cnn_model = model_loader.load_cnn_model(model_path)
    tfidf_vectorizer = model_loader.load_tfidf_vectorizer(tfidf_vectorizer_path)
    
    print("Loaded successfully")
    
    # Vectorize the text using TF-IDF vectorizer
    vectorized_text = tfidf_vectorizer.transform([preprocessed_text]).toarray()
    vectorized_text_reshaped = np.reshape(vectorized_text, (1, 1, vectorized_text.shape[1]))
    print("Vectorized text shape:", vectorized_text_reshaped.shape)
    print("Model input shape:", cnn_model.input_shape)

    if vectorized_text_reshaped.shape[-1] != cnn_model.input_shape[-1]:
        return JSONResponse(content={"error": "Input shape mismatch. Please ensure the TF-IDF vectorizer and model are compatible."})

    # Predict using the model
    predictions_prob = cnn_model.predict(vectorized_text_reshaped)
    predicted_class = np.argmax(predictions_prob, axis=1)[0]

    custom_predictions = (predictions_prob> 0.5).astype(int).flatten()
    
    data = {
        "predicted_class": str(custom_predictions[0]),
        "confidence": str(predictions_prob[0][0])
    }
    return JSONResponse(content=data)
@app.post("/api/v2/predict/text")
async def predict_hate_speech(input_text: InputText):
    preprocessed_text = preprocessed_text(input_text.text)
    
    model_loader = LoadModel()
    tfidf_vectorizer_path = "tfidf_vectorizer2.pkl"
    model_path = "cnn_model2.pkl"
    
    
    cnn_model = model_loader.load_cnn_model(model_path)
    tfidf_vectorizer = model_loader.load_tfidf_vectorizer(tfidf_vectorizer_path)
    
    print("Loaded succusfully")
    

    # Vectorize the text using TF-IDF vectorizer
    vectorized_text = tfidf_vectorizer.transform([preprocessed_text]).toarray()

    probability = cnn_model.predict(vectorized_text)
    
    custom_predictions = (probability> 0.5).astype(int).flatten()
    
    data = {
        "predicted_class": str(custom_predictions[0]),
        "confidence": str(probability[0][0])
    }
    return JSONResponse(content=data)



@app.get("/api/v1/audio/feature")
async def get_audio_feature(audio: UploadFile=File(...)):
    model_loader = LoadModel()
    audio_feature = np.array([model_loader.extract_features(audio)])
    data = {
        "name": "audio.wav",
        "confidence": str(audio_feature)
    }
    return JSONResponse(content=data)

@app.post("/api/v1/predict/audio")
async def predict_hate_speech(audio: UploadFile = File(...)):
    model_loader = LoadModel()
    model_path = "audio_model.pkl"
    
    print(audio.filename)
    
    path = 'audio.wav'
    
    with open(path, "wb") as f:
        f.write(audio.file.read())
    
    cnn_model = model_loader.load_audio_model(model_path)
    
    print("Loaded succusfully")
    
    cnn_model = model_loader.load_audio_model(model_path)
    
        
    audio_feature = np.array([model_loader.extract_features(path)])
    
    
    probability = cnn_model.predict(audio_feature)
    probability = probability.flatten()
    
    print("probability", probability)
    
    
    custom_predictions = np.argmax(probability)
    print("Custom prob:", custom_predictions)
    
    data = {
        "predicted_class": str(custom_predictions),
        "confidence": str(probability[custom_predictions])
    }
    print("data:", data)
    return JSONResponse(content=data)
    
@app.post("/api/v2/predict/audio")
async def predict_hate_speech(audio: UploadFile = File(...)):
    model_loader = LoadModel()
    model_path = "audio_model98.pkl"
    
    path = 'audio.wav'
    
    with open(path, "wb") as f:
        f.write(audio.file.read())
    
    cnn_model = model_loader.load_audio_model(model_path)
    
    print("Loaded succusfully 2")
    
    cnn_model = model_loader.load_audio_model(model_path)
    
    extracted_feature = extract_features(path)
    
    print(extracted_feature)
    data = pd.read_csv(StringIO(extracted_feature), delimiter=' ', header=None)
    print("data:", data)
    
    
    # Load the trained scaler for normalization
    loaded_scaler = joblib.load('scaler_model.pkl')

    # Extract the numeric columns from the test data

    # Use the loaded scaler to transform the test data
    normalized_data_test = pd.DataFrame(loaded_scaler.transform(data), columns=data.columns)
        
    # df_normalized_zscore = (data - data.mean(axis=0)) / data.std(axis=1)
    
    
    
    
    print("normalized data:", normalized_data_test)
    
    df_interpolated = normalized_data_test.interpolate()
    # audio_feature = np.array([df_interpolated])
    
    print("interpolated:", df_interpolated)
    
    
    probability = cnn_model.predict(df_interpolated)
    
    probability = probability.flatten()
    
    print("probability", probability)
    
    
    custom_predictions = np.argmax(probability)
    print("Custom prob:", custom_predictions)
    
    data = {
        "predicted_class": str(custom_predictions),
        "confidence": str(probability[custom_predictions])
    }
    print("data:", data)
    return JSONResponse(content=data)

@app.post("/api/v3/predict/audio")
async def predict_hate_speech(audio: UploadFile = File(...)):
    model_loader = LoadModel()
    model_path = "audio_model_comb.pkl"
    
    path = 'audio.wav'
    
    with open(path, "wb") as f:
        f.write(audio.file.read())
    
    cnn_model = model_loader.load_audio_model(model_path)
    
    print("Loaded succusfully 3")
    
    cnn_model = model_loader.load_audio_model(model_path)
    
    extracted_feature = extract_features(path)
    
    print(extracted_feature)
    data = pd.read_csv(StringIO(extracted_feature), delimiter=' ', header=None)
    print("data:", data)
    
    
    # Load the trained scaler for normalization
    loaded_scaler = joblib.load('scaler_model_comb.pkl')

    # Extract the numeric columns from the test data

    # Use the loaded scaler to transform the test data
    normalized_data_test = pd.DataFrame(loaded_scaler.transform(data), columns=data.columns)
        
    # df_normalized_zscore = (data - data.mean(axis=0)) / data.std(axis=1)
    
    
    
    
    print("normalized data:", normalized_data_test)
    
    df_interpolated = normalized_data_test.interpolate()
    # audio_feature = np.array([df_interpolated])
    
    print("interpolated:", df_interpolated)
    
    
    probability = cnn_model.predict(df_interpolated)
    
    probability = probability.flatten()
    
    print("probability", probability)
    
    
    custom_predictions = np.argmax(probability)
    print("Custom prob:", custom_predictions)
    
    data = {
        "predicted_class": str(custom_predictions),
        "confidence": str(probability[custom_predictions])
    }
    print("data:", data)
    return JSONResponse(content=data)

@app.post("/api/v3/newAudioApi")
async def newFunction(audio: UploadFile = File(...)):
    data = {
        "Message": "Received!",
    }
    return JSONResponse(content=data)

def preprocess_input(text):
    return text

def normalization(input_token):
    rep1 = re.sub('[ሃኅኃሐሓኻ]', 'ሀ', input_token)
    rep2 = re.sub('[ሑኁዅ]', 'ሁ', rep1)
    rep3 = re.sub('[ኂሒኺ]', 'ሂ', rep2)
    rep4 = re.sub('[ኌሔዄ]', 'ሄ', rep3)
    rep5 = re.sub('[ሕኅ]', 'ህ', rep4)
    rep6 = re.sub('[ኆሖኾ]', 'ሆ', rep5)
    rep7 = re.sub('[ሠ]', 'ሰ', rep6)
    rep8 = re.sub('[ሡ]', 'ሱ', rep7)
    rep9 = re.sub('[ሢ]', 'ሲ', rep8)
    rep10 = re.sub('[ሣ]', 'ሳ', rep9)
    rep11 = re.sub('[ሤ]', 'ሴ', rep10)
    rep12 = re.sub('[ሥ]', 'ስ', rep11)
    rep13 = re.sub('[ሦ]', 'ሶ', rep12)
    rep14 = re.sub('[ዓኣዐ]', 'አ', rep13)
    rep15 = re.sub('[ዑ]', 'ኡ', rep14)
    rep16 = re.sub('[ዒ]', 'ኢ', rep15)
    rep17 = re.sub('[ዔ]', 'ኤ', rep16)
    rep18 = re.sub('[ዕ]', 'እ', rep17)
    rep19 = re.sub('[ዖ]', 'ኦ', rep18)
    rep20 = re.sub('[ጸ]', 'ፀ', rep19)
    rep21 = re.sub('[ጹ]', 'ፁ', rep20)
    rep22 = re.sub('[ጺ]', 'ፂ', rep21)
    rep23 = re.sub('[ጻ]', 'ፃ', rep22)
    rep24 = re.sub('[ጼ]', 'ፄ', rep23)
    rep25 = re.sub('[ጽ]', 'ፅ', rep24)
    rep26 = re.sub('[ጾ]', 'ፆ', rep25)
    rep27 = re.sub('(ሉ[ዋአ])', 'ሏ', rep26)
    rep28 = re.sub('(ሙ[ዋአ])', 'ሟ', rep27)
    rep29 = re.sub('(ቱ[ዋአ])', 'ቷ', rep28)
    rep30 = re.sub('(ሩ[ዋአ])', 'ሯ', rep29)
    rep31 = re.sub('(ሱ[ዋአ])', 'ሷ', rep30)
    rep32 = re.sub('(ሹ[ዋአ])', 'ሿ', rep31)
    rep33 = re.sub('(ቁ[ዋአ])', 'ቋ', rep32)
    rep34 = re.sub('(ቡ[ዋአ])', 'ቧ', rep33)
    rep35 = re.sub('(ቹ[ዋአ])', 'ቿ', rep34)
    rep36 = re.sub('(ሁ[ዋአ])', 'ኋ', rep35)
    rep37 = re.sub('(ኑ[ዋአ])', 'ኗ', rep36)
    rep38 = re.sub('(ኙ[ዋአ])', 'ኟ', rep37)
    rep39 = re.sub('(ኩ[ዋአ])', 'ኳ', rep38)
    rep40 = re.sub('(ዙ[ዋአ])', 'ዟ', rep39)
    rep41 = re.sub('(ጉ[ዋአ])', 'ጓ', rep40)
    rep42 = re.sub('(ደ[ዋአ])', 'ዷ', rep41)
    rep43 = re.sub('(ጡ[ዋአ])', 'ጧ', rep42)
    rep44 = re.sub('(ጩ[ዋአ])', 'ጯ', rep43)
    rep45 = re.sub('(ጹ[ዋአ])', 'ጿ', rep44)
    rep46 = re.sub('(ፉ[ዋአ])', 'ፏ', rep45)
    rep47 = re.sub('[ቊ]', 'ቁ', rep46)  # ቁ can be written as ቊ
    rep48 = re.sub('[ኵ]', 'ኩ', rep47)  # ኩ can be also written as ኵ
    return rep48

def clean_text(text):
    # Normalize and clean text
    normalized_text = normalization(text)
    clean_text_var = re.sub(r'[^\w\s]', '', normalized_text)
    clean_text_var = re.sub(r'\d+', '', clean_text_var)
    clean_text_var = clean_text_var.strip()  # Strip leading and trailing spaces
    return clean_text_var
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)