# Utility functions for audio processing and inference
import numpy as np
import pywt
import librosa
import pandas as pd
import logging
import joblib
import soundfile as sf
import tempfile
import os
import subprocess
from pywt import WaveletPacket
from scipy.stats import entropy
import tensorflow as tf
tflite = tf.lite
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format='%(message)s')

# Load pre-fitted preprocessing artifacts
scaler = joblib.load('robust_scaler1.pkl')
transformer = joblib.load('yeo_johnson_transform.pkl')

# Model paths
MODEL_FLAC = "model1.tflite"  
MODEL_DEFAULT = "model.tflite"  

# Convert .flac to temporary .wav file
def convert_flac_to_wav(flac_path):
    try:
        data, sr = sf.read(flac_path)
        if data.ndim > 1:
            data = data.mean(axis=1)
        temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        sf.write(temp_wav.name, data, sr)
        return temp_wav.name
    except Exception as e:
        logging.error(f"Conversion failed for {flac_path}: {e}")
        return None

# Convert .webm to temporary .wav file
def convert_webm_to_wav(webm_path, duration=30):
    try:
        temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        command = [
            "ffmpeg",
            "-y",
            "-i", webm_path,
            "-t", str(duration),
            "-ac", "1",
            "-ar", "16000",
            temp_wav.name
        ]
        subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return temp_wav.name
    except Exception as e:
        logging.error(f"FFmpeg conversion failed for {webm_path}: {e}")
        return None

# Extract audio from mp4 using ffmpeg (first 30 seconds, mono, 16kHz)
def extract_audio_from_mp4(mp4_path, duration=5):  # <-- set to 3 seconds
    try:
        temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        command = [
            "ffmpeg",
            "-y",
            "-i", mp4_path,
            "-t", str(duration),
            "-ac", "1",
            "-ar", "16000",
            "-vn",
            temp_wav.name
        ]
        subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return temp_wav.name
    except Exception as e:
        logging.error(f"FFmpeg extraction failed for {mp4_path}: {e}")
        return None

# Load and normalize audio (handles .wav, .flac, .mp3, .m4a, .mp4)
def load_audio(filename):
    ext = os.path.splitext(filename)[1].lower()
    temp_wav_path = None

    if ext == ".mp4":
        temp_wav_path = extract_audio_from_mp4(filename, duration=30)
        if temp_wav_path is None:
            return None, None
        filename = temp_wav_path
    elif ext == ".webm":
        temp_wav_path = convert_webm_to_wav(filename, duration=30)
        if temp_wav_path is None:
            return None, None
        filename = temp_wav_path

    try:
        y, sr = librosa.load(filename, sr=16000)
        y = y / np.max(np.abs(y))
    except Exception as e:
        logging.error(f"Audio loading failed for {filename}: {e}")
        return None, None
    finally:
        if temp_wav_path:
            try:
                os.remove(temp_wav_path)
            except Exception:
                pass

    return sr, y

# Decomposition functions
def ewt_decompose(signal):
    return pywt.wavedec(signal, 'db4', level=4)

def wpt_decompose(signal, wavelet='db4', level=3):
    wp = WaveletPacket(data=signal, wavelet=wavelet, maxlevel=level)
    return [wp[node.path].data for node in wp.get_level(level)]

# Feature extraction
def extract_wavelet_features(ewt_coeffs, wpt_coeffs):
    features = []
    for coeff in ewt_coeffs:
        features += [np.mean(coeff), np.var(coeff), entropy(np.abs(coeff) + 1e-10)]
    for coeff in wpt_coeffs:
        features += [np.mean(coeff), np.var(coeff), entropy(np.abs(coeff) + 1e-10)]
    return features

def extract_spectral_feature(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc)
    if np.isclose(mfcc_mean, 0): # just to ensure any division by zero is avoided
        logging.warning("MFCC mean is zero, returning 0.0 for spectral feature")
        return 0.0
    return (1 - np.std(mfcc) / mfcc_mean)

def extract_pitch_variation(y, sr):
    pitches, mags = librosa.piptrack(y=y, sr=sr)
    valid = pitches[mags > np.median(mags)]
    if len(valid) == 0:
        return 0.0
    mean_valid = np.mean(valid)
    if mean_valid > 0:
        return np.std(valid) / mean_valid
    return 0.0

def extract_features(file_path):
    try:
        sr, y = load_audio(file_path)
        if y is None or len(y) == 0:
            raise ValueError("Invalid or empty audio")
        ewt = ewt_decompose(y)
        wpt = wpt_decompose(y)
        feats = extract_wavelet_features(ewt, wpt)
        spec_feat = extract_spectral_feature(y, sr)
        pitch_feat = extract_pitch_variation(y, sr)
        full_vector = [spec_feat, pitch_feat] + feats
        full_vector = np.array(full_vector, dtype=np.float32).reshape(1, -1)
        
        # Handle potential NaNs/Infs
        if np.any(np.isnan(full_vector)) or np.any(np.isinf(full_vector)):
            logging.warning(f"NaN/Inf found in features for {file_path}")
            full_vector = np.nan_to_num(full_vector, nan=0.0, posinf=0.0, neginf=0.0)
            
        return full_vector
    except Exception:
        logging.exception(f"❌ Feature extraction failed for {file_path}")
        return None

# Inference
def run_inference(file_path):
    try:
        # Determine model based on file extension
        ext = os.path.splitext(file_path)[1].lower()
        if ext in [".flac", ".mp4"]:
            model_path = MODEL_FLAC
            model_used = "FLAC/ASVspoof/Video-specialized"
        else:
            model_path = MODEL_DEFAULT
            model_used = "ADD/Audio-specialized"
        
        interpreter = tflite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        features = extract_features(file_path)
        if features is None:
            return {"status": "error", "message": "Feature extraction failed"}

        features = transformer.transform(features)
        if hasattr(scaler, 'feature_names_in_'):
            df = pd.DataFrame(features, columns=scaler.feature_names_in_)
            features_scaled = scaler.transform(df)
        else:
            features_scaled = scaler.transform(features)

        features_scaled = features_scaled.astype(np.float32)
        if features_scaled.shape[1] != input_details[0]['shape'][1]:
            return {"status": "error", "message": "Input shape mismatch with model"}

        interpreter.set_tensor(input_details[0]['index'], features_scaled)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])

        prob = (output[0][0])

        # Custom prediction logic:
        # If the original file was .mp4 (video), invert the prediction logic
        if ext == ".mp4":
            pred = int(prob <= 0.5)
            label = "Bona fide" if pred else "Spoof"
            confidence = round((1 - prob) if pred else prob, 4)
        else:
            pred = int(prob > 0.5)
            label = "Bona fide" if pred else "Spoof"
            confidence = round(prob if pred else 1 - prob, 4)

        return {
            "file": file_path,
            "status": "success",
            "label": label,
            "prediction": pred,
            "confidence": confidence,
            "model_used": model_used
        }
    except Exception as e:
        logging.exception("❌ Inference failed")
        return {"status": "error", "message": f"Inference error: {str(e)}"}
