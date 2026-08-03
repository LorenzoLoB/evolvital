
import vita
import numpy as np
import librosa
from dtaidistance import dtw_ndim
import scipy
import matplotlib.pyplot as plt
import wave
from scipy.io import wavfile
import math
from scipy.spatial.distance import cosine

def render_synth(synth):
    audio = synth.render(PITCH, VELOCITY, NOTE_DUR, RENDER_DUR)
    if (audio.ndim == 2):
        audio = np.mean(audio, axis=0)
    audio = audio.astype(np.float32)
    return audio

def graph_spec(melSG,i):
    #melSG = librosa.power_to_db(melSG, ref=10**-2, amin=10**-2)
    plt.figure(figsize=(10,4))
    librosa.display.specshow(
        melSG,
        sr=SAMPLE_RATE,
        hop_length=1024,
        x_axis='time',
        y_axis='mel'
    )
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Mel Spectrogram of {i}')
    plt.tight_layout()
    plt.savefig(f"spectrograms/{i}.png")
    plt.close()

def create_spec(audio):
    k_window = scipy.signal.windows.kaiser(M=8192,beta=20)
    spec = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_mels=256,
        hop_length=1024,
        window=k_window,
        n_fft=8192,
        fmin=0,
        fmax=15000
    )
    spec = spec.astype(np.float32)      #???
    return spec

def create_mel_spec(audio):
    spec = create_spec(audio)
    mel_spec = librosa.power_to_db(spec, ref=10**-2, amin=10**-2)
    return mel_spec


def extract_mfcc(audio):
    mfcc = librosa.feature.mfcc(
            y=audio,
            sr=SAMPLE_RATE,
            n_mfcc=MFCC_COUNT
        )
    return mfcc


def dtw_prep(audio, ref_mean=None, ref_std=None):
    mfcc = extract_mfcc(audio).T[:, 1:]
    if (ref_mean == None):
        ref_mean = mfcc.mean(axis=0, keepdims=True)
        ref_std = mfcc.std(axis=0, keepdims=True)
        ref_std[std < 1e-8] = 1e-8
        a = np.ascontiguousarray((mfcc - ref_mean) / ref_std, dtype=np.float64)
        return a, ref_mean, ref_std
    return a

def dtw_distance(A, B):
    raw_distance = dtw_ndim.distance_fast(A,B)
    distance = raw_distance /(len(A) + len(B))
    similarity = 1.0 / (1.0 + distance)

    return similarity

def mfcc_mean(audio):
    
    mfcc = extract_mfcc(audio)
    mean = np.mean(mfcc, axis=1)
    return mean

def mfcc_distance(mfccA, mfccB):
    similarity = 1 - cosine(mfccA, mfccB)
    return similarity

def centroid_distance(centroidA, centroidB):
    distance = np.linalg.norm(centroidA - centroidB)
    score = np.exp(-distance)
    return score

def spectral_centroid(audio):
    centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=2048,
        hop_length=512,
        win_length=2048,
        window="hann",
        center=True
    )
    centroid /= SAMPLE_RATE/2
    return centroid

MFCC_COUNT = 20
SAMPLE_RATE = 44_100
BPM = 120.0
NOTE_DUR = 2#1.2
RENDER_DUR = 3#1.41
PITCH = 48
VELOCITY = 0.7

synth = vita.Synth()
controls = synth.get_controls()
synth.load_preset("targets/target_3.vital")
x = controls["lfo_1_frequency"].get_normalized()


audioA = render_synth(synth)



synth2 = vita.Synth()
controls2 = synth2.get_controls()
synth2.load_preset("targets/target_3.vital")
controls2["lfo_1_frequency"].set_normalized(x*2)


audioB = render_synth(synth2)
A = dtw_prep(audioA)
B = dtw_prep(audioB)

similarity = dtw_distance(A, B)
similarity = mfcc_distance(mfcc_mean(audioA), mfcc_mean(audioB))
similarity = centroid_distance(spectral_centroid(audioA), spectral_centroid(audioB))

print(similarity)

print (audioA==audioB)

graph_spec(create_mel_spec(audioA), "A")
graph_spec(create_mel_spec(audioB), "B")

wavfile.write("audio/A", SAMPLE_RATE, audioA)

wavfile.write("audio/B", SAMPLE_RATE, audioB)