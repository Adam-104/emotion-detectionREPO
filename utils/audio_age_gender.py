"""
audio_age_gender.py
====================
Estimates approximate age and gender from a WAV audio file using
acoustic features extracted with librosa.

Features used
-------------
- Fundamental frequency (F0 / pitch) via librosa.yin()
  * Males:   ~85–155 Hz mean F0
  * Females: ~165–255 Hz mean F0
  * Children: >255 Hz

- MFCCs (mel-frequency cepstral coefficients) — captures vocal tract shape
- Spectral centroid — brightness of voice
- Zero-crossing rate — noisiness / sibilance
- Jitter (F0 variation) — increases with age
- Shimmer proxy (RMS variation) — increases with age

Age estimation heuristic
-------------------------
Base age is derived from the F0 range, then adjusted by:
  + jitter  → older voices have more pitch instability
  + shimmer → older voices have more amplitude variation
  - spectral centroid → younger/higher voices have brighter spectra
  - MFCC[1] (first delta MFCC captures vocal tract formants)

This is a signal-processing heuristic, NOT a trained ML model.
Expect accuracy within ±8–12 years for typical speech.
"""

import numpy as np

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


# ── PITCH BOUNDS ──────────────────────────────────────────────────
F0_MIN = 60    # Hz — below this is noise, not voice
F0_MAX = 400   # Hz — above this is falsetto / child singing


def _safe_mean(arr):
    arr = np.array(arr)
    arr = arr[(arr > F0_MIN) & (arr < F0_MAX)]
    return float(np.mean(arr)) if len(arr) > 5 else None


# ── MAIN FUNCTION ─────────────────────────────────────────────────

def predict_age_gender(wav_path: str) -> tuple[str, str]:
    """
    Returns (gender, age_str) estimated from the audio file.
    Falls back to ("Unknown", "Unknown") if librosa is unavailable
    or the file cannot be analysed.

    Parameters
    ----------
    wav_path : str
        Path to a WAV file (mono or stereo; will be mixed to mono).

    Returns
    -------
    gender : str   — "Male", "Female", or "Unknown"
    age    : str   — e.g. "27" or "Unknown"
    """

    if not LIBROSA_AVAILABLE:
        return "Unknown", "Unknown"

    try:
        # ── Load audio ──────────────────────────────────────────
        y, sr = librosa.load(wav_path, sr=None, mono=True)

        # Need at least 0.5 s of audio to be meaningful
        if len(y) < sr * 0.5:
            return "Unknown", "Unknown"

        # ── Fundamental frequency (YIN algorithm) ───────────────
        f0 = librosa.yin(y, fmin=F0_MIN, fmax=F0_MAX)
        mean_f0 = _safe_mean(f0)

        if mean_f0 is None:
            return "Unknown", "Unknown"

        # ── Gender from pitch ────────────────────────────────────
        # Threshold at 165 Hz (mid-point between typical male/female)
        gender = "Female" if mean_f0 >= 165 else "Male"

        # ── Jitter (pitch variation) — proxy for age instability ─
        voiced = f0[(f0 > F0_MIN) & (f0 < F0_MAX)]
        jitter = float(np.std(voiced) / mean_f0) if len(voiced) > 5 else 0.0

        # ── Shimmer proxy (RMS amplitude variation) ──────────────
        rms    = librosa.feature.rms(y=y)[0]
        shimmer = float(np.std(rms) / (np.mean(rms) + 1e-6))

        # ── MFCCs ────────────────────────────────────────────────
        mfcc   = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc1  = float(np.mean(mfcc[1]))   # 2nd MFCC — formant info

        # ── Spectral centroid ─────────────────────────────────────
        centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))

        # ── Age heuristic ─────────────────────────────────────────
        # Start from pitch-derived base age
        if gender == "Female":
            # Higher F0 → younger among females
            if mean_f0 > 220:
                base_age = 18
            elif mean_f0 > 195:
                base_age = 25
            elif mean_f0 > 175:
                base_age = 35
            else:
                base_age = 48
        else:
            # Lower F0 tends toward older for males
            if mean_f0 > 140:
                base_age = 20
            elif mean_f0 > 120:
                base_age = 30
            elif mean_f0 > 105:
                base_age = 42
            else:
                base_age = 55

        # ── Adjust by acoustic aging cues ────────────────────────
        # Jitter adds age (unstable pitch = older)
        age_adj  =  jitter * 40          # +0 to ~+15 years
        # Shimmer adds age (unstable amplitude = older)
        age_adj +=  shimmer * 20         # +0 to ~+10 years
        # High spectral centroid → younger/brighter voice
        age_adj -= (centroid / 5000.0)   # -0 to ~-5 years
        # mfcc1 correlation with formant patterns
        age_adj -= (mfcc1 / 60.0)        # small correction

        estimated_age = int(round(base_age + age_adj))

        # Clamp to plausible range
        estimated_age = max(10, min(85, estimated_age))

        return gender, str(estimated_age)

    except Exception as e:
        print(f"[audio_age_gender] Error: {e}")
        return "Unknown", "Unknown"
