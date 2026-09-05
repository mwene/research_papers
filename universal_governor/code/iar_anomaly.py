"""
UNIVERSAL IAR ANOMALY DETECTION ENGINE
Processes any data stream (bytes, video, text, sensor, audio)
Detects anomalies in any phenomenon using the IAR framework.

Core insight: anomalies are IAR regime shifts. When the IAR parameters
(beta, gamma, lambda, eta, alpha, theta) deviate from their normal range,
the system is entering a new fate -- internal equilibrium, dominance, or
dissolution.

Author: Macharia Barii
Dependencies: numpy, scipy.
"""

import numpy as np
import hashlib
from scipy.fft import fft, fftfreq
from scipy.stats import entropy, kurtosis, skew
from scipy.signal import find_peaks
from collections import deque
import warnings
warnings.filterwarnings("ignore")


def _grayscale(frame):
    """RGB/any (H, W, C) or (H, W) frame -> grayscale float (numpy only)."""
    a = np.asarray(frame, dtype=float)
    if a.ndim == 3:
        if a.shape[-1] == 3:
            return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
        return a.mean(axis=-1)
    return a


def _gradient_magnitude(gray):
    """Sobel-like edge magnitude without OpenCV."""
    gx = np.gradient(gray, axis=1)
    gy = np.gradient(gray, axis=0)
    return np.hypot(gx, gy)


# ============================================================
# 1. UNIFIED FEATURE EXTRACTOR
# ============================================================

class UniversalFeatureExtractor:
    """Extracts features from ANY data stream type.

    Converts raw data (bytes, video, text, sensor, audio) to
    IAR-compatible numerical features.
    """

    def __init__(self, feature_dim=64):
        self.feature_dim = feature_dim
        self.prev_frame = None

    def extract(self, data, data_type="sensor"):
        if data_type == "bytes":
            return self._extract_bytes(data)
        elif data_type == "video":
            return self._extract_video(data)
        elif data_type == "text":
            return self._extract_text(data)
        elif data_type == "sensor":
            return self._extract_sensor(data)
        elif data_type == "audio":
            return self._extract_audio(data)
        return self._extract_custom(data)

    def _pad(self, features):
        features = np.asarray(features, dtype=float)
        features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
        if len(features) < self.feature_dim:
            return np.pad(features, (0, self.feature_dim - len(features)))
        return features[:self.feature_dim]

    def _extract_bytes(self, data):
        """Byte-stream features: statistics, entropy, spectral, periodic."""
        arr = np.frombuffer(data, dtype=np.uint8) if isinstance(data, bytes) else np.asarray(data)
        arr = np.asarray(arr, dtype=float)
        if arr.size == 0:
            return np.zeros(self.feature_dim)

        mean = np.mean(arr); std = np.std(arr)
        skewness = skew(arr); kurt = kurtosis(arr)
        hist, _ = np.histogram(arr, bins=64)
        hist = hist / (arr.size + 1e-8)
        entropy_val = entropy(hist + 1e-8)

        if arr.size > 4:
            power = np.abs(fft(arr))
            freqs = fftfreq(arr.size)
            dominant_freq = freqs[np.argmax(power[1:]) + 1]
            spectral_centroid = np.sum(freqs * power) / (np.sum(power) + 1e-8)
        else:
            dominant_freq = 0; spectral_centroid = 0

        if arr.size > 10:
            acf = np.correlate(arr - mean, arr - mean, mode="full")
            acf = acf[len(acf) // 2:] / (std**2 * arr.size + 1e-8)
            peaks, _ = find_peaks(acf, height=0.3)
            periodicity = len(peaks) / len(acf)
        else:
            periodicity = 0

        return self._pad([mean, std, skewness, kurt, entropy_val,
                          np.real(dominant_freq), spectral_centroid, periodicity,
                          np.percentile(arr, 25), np.percentile(arr, 75),
                          np.max(arr), np.min(arr)])

    def _extract_video(self, frame):
        """Video-frame features: grayscale stats, edges, motion, 2D FFT."""
        if isinstance(frame, bytes):
            nparr = np.frombuffer(frame, dtype=np.uint8)
            side = int(np.sqrt(nparr.size // 3)) or 1
            need = side * side * 3
            img = nparr[:need].reshape(side, side, 3).astype(float)
        else:
            img = frame
        try:
            gray = _grayscale(img)
        except Exception:
            return np.zeros(self.feature_dim)
        if gray.size == 0:
            return np.zeros(self.feature_dim)

        mean = np.mean(gray); std = np.std(gray)
        skewness = skew(gray.flatten()); kurt = kurtosis(gray.flatten())
        entropy_val = entropy(np.histogram(gray, bins=64)[0] + 1e-8)
        edge_density = np.mean(_gradient_magnitude(gray) > np.std(gray) + 1e-8)

        if self.prev_frame is not None and self.prev_frame.shape == gray.shape:
            flow_mag = np.hypot(*np.gradient(gray - self.prev_frame))
            motion_mag_mean = np.mean(flow_mag)
            motion_ang_std = np.std(np.angle(np.gradient(gray)[0] + 1j * np.gradient(gray)[1]))
        else:
            motion_mag_mean = 0; motion_ang_std = 0
        self.prev_frame = gray

        power = np.abs(np.fft.fft2(gray))
        fx = np.fft.fftfreq(gray.shape[0]).reshape(-1, 1)
        fy = np.fft.fftfreq(gray.shape[1]).reshape(1, -1)
        sx = np.sum(fx * power) / (np.sum(power) + 1e-8)
        sy = np.sum(fy * power) / (np.sum(power) + 1e-8)

        return self._pad([mean, std, skewness, kurt, entropy_val,
                          edge_density, motion_mag_mean, motion_ang_std,
                          np.real(sx), np.real(sy),
                          np.percentile(gray, 25), np.percentile(gray, 75)])

    def _extract_text(self, text):
        """Text features: word/sentence stats, lexical diversity, punctuation."""
        if isinstance(text, bytes):
            text = text.decode("utf-8", errors="ignore")
        words = text.split()
        sentences = [s for s in text.split(".") if len(s) > 0]
        word_lengths = [len(w) for w in words]
        sentence_lengths = [len(s.split()) for s in sentences]

        return self._pad([len(words), len(sentences),
                          np.mean(word_lengths) if word_lengths else 0,
                          np.std(word_lengths) if word_lengths else 0,
                          np.mean(sentence_lengths) if sentence_lengths else 0,
                          np.std(sentence_lengths) if sentence_lengths else 0,
                          len(set(words)) / (len(words) + 1e-8),
                          text.count("!"), text.count("?"), text.count(","), text.count('"'),
                          np.std([len(s) for s in sentences]) if sentences else 0])

    def _extract_sensor(self, data):
        """Sensor time-series features: stats, trend, spectral, rate of change."""
        if isinstance(data, (int, float)):
            arr = np.array([data])
        elif isinstance(data, bytes):
            arr = np.frombuffer(data, dtype=np.float32)
        else:
            arr = np.asarray(data, dtype=float).flatten()
        arr = arr[np.isfinite(arr)]
        if arr.size == 0 or arr.size < 2:
            return np.zeros(self.feature_dim)

        mean = np.mean(arr); std = np.std(arr)
        skewness = skew(arr); kurt = kurtosis(arr)
        coeffs = np.polyfit(np.arange(arr.size), arr, 1)
        trend = coeffs[0]

        if arr.size > 10:
            power = np.abs(fft(arr))
            freqs = fftfreq(arr.size)
            dominant_freq = freqs[np.argmax(power[1:]) + 1]
            power_norm = power / (np.sum(power) + 1e-8)
            spectral_entropy = entropy(power_norm + 1e-8)
            peaks, _ = find_peaks(arr, height=np.mean(arr) + np.std(arr))
            n_peaks = len(peaks)
        else:
            dominant_freq = 0; spectral_entropy = 0; n_peaks = 0

        diff = np.diff(arr)
        rate_mean = np.mean(diff) if diff.size > 0 else 0
        rate_std = np.std(diff) if diff.size > 0 else 0

        return self._pad([mean, std, skewness, kurt, trend,
                          np.real(dominant_freq), spectral_entropy, n_peaks,
                          rate_mean, rate_std,
                          np.percentile(arr, 25), np.percentile(arr, 75)])

    def _extract_audio(self, data):
        """Audio-data features (spectral + temporal)."""
        return self._extract_sensor(data)

    def _extract_custom(self, data):
        try:
            return self._extract_sensor(np.array(data))
        except Exception:
            return np.zeros(self.feature_dim)


# ============================================================
# 2. IAR STATE ESTIMATOR
# ============================================================

class IARStateEstimator:
    """Estimates IAR parameters from feature streams.

    Uses online learning to adapt to normal behavior (Pattern of Life).
    """

    def __init__(self, feature_dim=64, window_size=100):
        self.feature_dim = feature_dim
        self.window_size = window_size

        self.X_C = np.zeros(feature_dim)   # Action
        self.X_O = np.zeros(feature_dim)   # Open state
        self.R = np.zeros(feature_dim)     # Reaction

        self.theta = {"beta": 0.5, "theta": 1.0, "gamma": 0.5,
                      "lam": 0.1, "eta": 0.3, "alpha": 0.5}

        self.history = deque(maxlen=window_size)
        self.param_history = deque(maxlen=window_size)

        self.baseline = None
        self.baseline_std = None
        self.baseline_count = 0
        self.threshold = 3.0  # standard deviations

    def update(self, features):
        """Update IAR state with a new feature vector."""
        features = np.asarray(features, dtype=float)
        prev_X_C = self.X_C.copy(); prev_X_O = self.X_O.copy(); prev_R = self.R.copy()

        J = self.theta["beta"] * (self.X_C - self.theta["theta"] * self.X_O)
        self.X_C = features
        self.X_O = 0.9 * self.X_O + 0.1 * features
        diff = self.X_C - prev_X_C
        self.R = (self.theta["gamma"] * diff - self.theta["lam"] * self.R
                  - self.theta["eta"] * self.X_O * self.R)

        self._learn_parameters(prev_X_C, prev_X_O, prev_R)
        anomaly_score = self._compute_anomaly(features)

        self.history.append({"X_C": self.X_C.copy(), "X_O": self.X_O.copy(),
                             "R": self.R.copy(), "J": J, "features": features.copy()})
        self.param_history.append(self.theta.copy())
        self._update_baseline(features)
        fate = self._classify_fate()

        return {"theta": self.theta.copy(), "anomaly_score": anomaly_score,
                "fate": fate, "X_C": self.X_C.copy(), "X_O": self.X_O.copy(),
                "R": self.R.copy(), "J": J}

    def _learn_parameters(self, prev_X_C, prev_X_O, prev_R, lr=0.001):
        """Online learning of IAR parameters via finite differences."""
        for param in ["beta", "theta", "gamma", "lam", "eta", "alpha"]:
            orig = self.theta[param]
            grad = 0.0
            for delta in [0.001, -0.001]:
                self.theta[param] = orig + delta
                pred_R = (self.theta["gamma"] * (self.X_C - prev_X_C)
                          - self.theta["lam"] * prev_R
                          - self.theta["eta"] * self.X_O * prev_R)
                loss = (np.mean((self.X_C - prev_X_C
                                 - self.theta["beta"] * (self.X_C - self.theta["theta"] * self.X_O))**2)
                        + np.mean((pred_R - self.R)**2))
                grad += loss / (2 * delta)
            self.theta[param] = orig - lr * np.clip(grad, -1, 1)

    def _compute_anomaly(self, features):
        """Multi-factor anomaly score: parameter + stability + feature deviation."""
        param_deviation = 0
        if self.baseline is not None:
            param_deviation = np.mean([
                (self.theta["beta"] - self.baseline["beta"]) / (self.baseline_std["beta"] + 1e-8),
                (self.theta["gamma"] - self.baseline["gamma"]) / (self.baseline_std["gamma"] + 1e-8),
                (self.theta["lam"] - self.baseline["lam"]) / (self.baseline_std["lam"] + 1e-8)])

        bg = self.theta["beta"] * self.theta["gamma"]
        stability_score = np.abs(bg - self.theta["lam"] - 0.5) / 0.5

        feature_dev = 0
        if self.baseline is not None:
            feature_dev = np.mean(np.abs(features - self.baseline_features)
                                  / (self.baseline_features_std + 1e-8))

        return 0.3 * param_deviation + 0.3 * stability_score + 0.4 * feature_dev

    def _update_baseline(self, features):
        """Update the Pattern of Life baseline (exponential moving average)."""
        if self.baseline_count < 100:
            self.baseline_count += 1
        if self.baseline is None:
            self.baseline = self.theta.copy()
            self.baseline_features = features.copy()
            self.baseline_std = {k: 0.1 for k in self.theta}
            self.baseline_features_std = np.ones_like(features) * 0.1
        else:
            a = 0.05
            for key in self.theta:
                self.baseline[key] = (1 - a) * self.baseline[key] + a * self.theta[key]
                self.baseline_std[key] = ((1 - a) * self.baseline_std[key]
                                          + a * np.abs(self.theta[key] - self.baseline[key]))
            self.baseline_features = (1 - a) * self.baseline_features + a * features
            self.baseline_features_std = ((1 - a) * self.baseline_features_std
                                          + a * np.abs(features - self.baseline_features))

    def _classify_fate(self):
        """Classify the four fates based on IAR parameters."""
        beta = self.theta["beta"]; gamma = self.theta["gamma"]
        lam = self.theta["lam"]; alpha = self.theta["alpha"]; eta = self.theta["eta"]
        bg = beta * gamma
        if bg < lam * 0.5:
            return "Internal Equilibrium"
        elif bg < lam:
            return "Joint Equilibrium"
        elif alpha > 0.6:
            return "C Dominates"
        elif alpha < 0.4:
            return "O Dominates"
        elif eta > 0.3:
            return "Dissolution"
        return "Transitional"

    def get_pattern_of_life(self):
        if self.baseline is None:
            return None
        return {"baseline_params": self.baseline,
                "baseline_params_std": self.baseline_std,
                "baseline_features": self.baseline_features,
                "baseline_features_std": self.baseline_features_std,
                "threshold": self.threshold}


# ============================================================
# 3. UNIVERSAL ANOMALY DETECTION ENGINE
# ============================================================

class IARAnomalyDetector:
    """Universal IAR-based anomaly detection engine.

    Processes ANY data stream and detects anomalies in ANY phenomenon.
    """

    def __init__(self, feature_dim=64, window_size=100):
        self.feature_dim = feature_dim
        self.window_size = window_size
        self.feature_extractor = UniversalFeatureExtractor(feature_dim)
        self.state_estimator = IARStateEstimator(feature_dim, window_size)
        self.anomaly_log = []
        self.alert_callbacks = []
        self._seen = 0

    def process(self, data, data_type="sensor"):
        """Process a single data point (streaming)."""
        features = self.feature_extractor.extract(data, data_type)
        state = self.state_estimator.update(features)
        is_anomaly = state["anomaly_score"] > self.state_estimator.threshold
        self._seen += 1
        if is_anomaly:
            entry = {
                "timestamp": self._seen,
                "features": features.tolist(),
                "state": {"anomaly_score": state["anomaly_score"],
                          "fate": state["fate"], "theta": state["theta"]},
                "data_type": data_type,
                "data_hash": hashlib.md5(str(data).encode()).hexdigest()[:8]}
            self.anomaly_log.append(entry)
            for callback in self.alert_callbacks:
                callback(state, features, data)
        return {"state": state, "is_anomaly": is_anomaly,
                "features": features, "fate": state["fate"],
                "pattern_of_life": self.state_estimator.get_pattern_of_life()}

    def process_batch(self, data_batch, data_type="sensor"):
        return [self.process(d, data_type) for d in data_batch]

    def add_alert_callback(self, callback):
        self.alert_callbacks.append(callback)

    def get_anomaly_report(self):
        if len(self.anomaly_log) == 0:
            return "No anomalies detected"
        fates = [a["state"]["fate"] for a in self.anomaly_log]
        return {
            "total_anomalies": len(self.anomaly_log),
            "anomaly_rate": len(self.anomaly_log) / max(1, self._seen),
            "most_common_fate": max(set(fates), key=fates.count),
            "anomaly_events": self.anomaly_log[-10:],
        }


USE_CASES = {
    "Cybersecurity": {"data_type": "bytes",
        "anomalies": ["malware", "intrusion", "data exfiltration"],
        "IAR_signature": "C dominates (attacker controls system)"},
    "Healthcare": {"data_type": "sensor",
        "anomalies": ["cardiac arrhythmia", "seizure", "apnea"],
        "IAR_signature": "Dissolution (system breakdown)"},
    "Industrial IoT": {"data_type": "sensor",
        "anomalies": ["machine failure", "quality deviation", "process upset"],
        "IAR_signature": "Transitional -> Dissolution"},
    "Financial": {"data_type": "sensor",
        "anomalies": ["market crash", "fraud", "regime change"],
        "IAR_signature": "O dominates (market forces)"},
    "Video Surveillance": {"data_type": "video",
        "anomalies": ["intrusion", "violence", "abandoned object"],
        "IAR_signature": "Internal Equilibrium -> C dominates"},
    "Social Media": {"data_type": "text",
        "anomalies": ["misinformation", "hate speech", "viral spread"],
        "IAR_signature": "O dominates (social contagion)"},
}


def run_use_case(use_case):
    """Run a specific anomaly-detection use case."""
    uc = USE_CASES[use_case]
    print(f"\n{'='*60}\nUSE CASE: {use_case.upper()}\n{'='*60}")
    print(f" Data Type: {uc['data_type']}")
    print(f" Anomalies: {', '.join(uc['anomalies'])}")
    print(f" IAR Signature: {uc['IAR_signature']}")

    detector = IARAnomalyDetector(feature_dim=32, window_size=50)
    dtype = uc["data_type"]

    if dtype == "bytes":
        normal = [bytes(np.random.randint(0, 255, 100).astype(np.uint8)) for _ in range(100)]
        anomalies = [b"\x00" * 100, b"\xFF" * 100, b"\x90" * 100 + b"\xE9" * 100]
    elif dtype == "sensor":
        normal = [10 + 5 * np.sin(0.1 * i) + np.random.randn() * 1 for i in range(100)]
        anomalies = [50 + 20 * np.random.randn(), -5 + np.random.randn(), 10 + 30 * np.random.randn()]
    elif dtype == "text":
        normal = ["Normal text message" for _ in range(100)]
        anomalies = ["!!! EMERGENCY !!!", "ALERT: SYSTEM BREACH", "!" * 100]
    else:  # video
        normal = [np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(30)]
        anomalies = [np.zeros((64, 64, 3), dtype=np.uint8),
                     np.ones((64, 64, 3), dtype=np.uint8) * 255]

    for d in normal:
        detector.process(d, dtype)
    for d in anomalies:
        r = detector.process(d, dtype)
        print(f" Anomaly candidate: Fate={r['fate']}, "
              f"Score={r['state']['anomaly_score']:.3f}")
    return detector


if __name__ == "__main__":
    for uc in ["Cybersecurity", "Healthcare", "Industrial IoT"]:
        run_use_case(uc)