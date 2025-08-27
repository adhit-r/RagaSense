import numpy as np
import librosa
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models, initializers
import tensorflow as tf
import os

# --- Feature Extraction ---

ROW_DURATION = 10  # seconds
SAMPLING_RATE = 22050
FRAME_SIZE = 2048
HOP = 2048
SAMPLE_ROW_LEN = ROW_DURATION * SAMPLING_RATE
JUMP = SAMPLING_RATE * 5  # 5s hop
BINS = librosa.fft_frequencies(sr=SAMPLING_RATE, n_fft=FRAME_SIZE)


def convert_to_spec(row, frame_size=FRAME_SIZE, hop=HOP):
    stft = librosa.stft(row, n_fft=frame_size, hop_length=hop, center=False)
    spec = np.abs(stft)
    return spec


def create_x_row(signal, sample_row_len=SAMPLE_ROW_LEN, rate=SAMPLING_RATE, jump=JUMP, bins=BINS):
    """
    Segment audio and extract max-counts frequency vector for each segment.
    """
    rows = []
    for i in range(0, len(signal) - sample_row_len, jump):
        r = signal[i:i + sample_row_len]
        r_spec = convert_to_spec(r)
        max_f = r_spec.argmax(axis=0)
        count_r = np.zeros(len(bins))
        for j in max_f:
            count_r[j] += 1
        rows.append(count_r)
    return rows


def extract_features_from_file(filepath):
    signal, _ = librosa.load(filepath, sr=SAMPLING_RATE)
    return create_x_row(signal)

# --- Model Building ---

def build_dense_nn(input_dim, num_classes):
    input_seq = layers.Input(shape=(input_dim,), dtype='float32')
    x = layers.Dense(64, kernel_initializer=initializers.RandomUniform(), activation='relu')(input_seq)
    x = layers.Dense(32, kernel_initializer=initializers.RandomUniform(), activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(32, kernel_initializer=initializers.RandomUniform(), activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(32, kernel_initializer=initializers.RandomUniform(), activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    if num_classes == 2:
        preds = layers.Dense(1, activation="sigmoid")(x)
        model = models.Model(input_seq, preds)
        model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.SGD(learning_rate=0.01), metrics=['accuracy'])
    else:
        preds = layers.Dense(num_classes, activation="softmax")(x)
        model = models.Model(input_seq, preds)
        model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.SGD(learning_rate=0.01), metrics=['accuracy'])
        return model
    
# --- Classical ML Models ---
def train_logistic_regression(X, y):
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X, y)
    return lr

def train_svm(X, y):
    svm = SVC(probability=True)
    svm.fit(X, y)
    return svm

def train_xgboost(X, y):
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    xgb.fit(X, y)
    return xgb

# --- Inference Utilities ---
def predict_segments(model, segments, is_keras=True):
    preds = []
    for seg in segments:
        seg = seg.reshape(1, -1)
        if is_keras:
            p = model.predict(seg)
            if p.shape[-1] == 1:
                preds.append(int(p[0][0] > 0.5))
            else:
                preds.append(np.argmax(p[0]))
        else:
            p = model.predict(seg)
            preds.append(int(p[0]))
    # Majority voting
    return int(np.median(preds))

# --- Save/Load Models ---
def save_keras_model(model, path):
    model.save(path)

def load_keras_model(path):
    return models.load_model(path)

def save_sklearn_model(model, path):
    import pickle
    with open(path, 'wb') as f:
        pickle.dump(model, f)

def load_sklearn_model(path):
    import pickle
    with open(path, 'rb') as f:
        return pickle.load(f)

# --- Main Classifier API ---
class RagaClassifier:
    def __init__(self, model_type='dense', model_path=None, num_classes=2):
        self.model_type = model_type
        self.model = None
        self.num_classes = num_classes
        if model_path:
            if model_type == 'dense':
                self.model = load_keras_model(model_path)
            else:
                self.model = load_sklearn_model(model_path)

    def train(self, X, y):
        if self.model_type == 'dense':
            self.model = build_dense_nn(X.shape[1], self.num_classes)
            self.model.fit(X, y, batch_size=32, epochs=100, validation_split=0.2)
        elif self.model_type == 'logreg':
            self.model = train_logistic_regression(X, y)
        elif self.model_type == 'svm':
            self.model = train_svm(X, y)
        elif self.model_type == 'xgb':
            self.model = train_xgboost(X, y)
        else:
            raise ValueError('Unknown model type')

    def predict(self, segments):
        is_keras = self.model_type == 'dense'
        return predict_segments(self.model, segments, is_keras=is_keras)

    def save(self, path):
        if self.model_type == 'dense':
            save_keras_model(self.model, path)
        else:
            save_sklearn_model(self.model, path)

    @staticmethod
    def extract_features(filepath):
        return extract_features_from_file(filepath) 