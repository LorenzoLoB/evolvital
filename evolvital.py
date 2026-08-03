import kymatio
import kymatio.torch as kt

#print(kymatio.__version__)
#print(dir(kt))

from scipy.io import wavfile
import vita
import random as r
from dataclasses import dataclass
import math
#from assist import change_version
import time
import numpy as np
import librosa

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import statistics as stat
import scipy
from scipy.spatial.distance import cosine
import pygad
from tqdm import tqdm
import json
import sys
import torch
from kymatio.torch import TimeFrequencyScattering
from dtaidistance import dtw_ndim
import os
from datetime import datetime




@dataclass
class Parameter:
    name: str
    value: float
    min: float
    max: float



#=============================================================================+
#____________________________HELPER FUNCTIONS________________________________
#=============================================================================+

def change_version(file):
    with open(file, "r") as f:
        data = json.load(f)

    data['synth_version'] = "1.5.5"
    with open(file, "w") as f:
        json.dump(data,f,indent=2)

def format_time(seconds):
    minutes = int(seconds // 60)
    remaining_seconds = int(seconds % 60)
    return f"{minutes} min {remaining_seconds} s"

def update_bar():
    pbar.update(1)

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

    return melSG

def create_random_genome(parameter_base, mod_connections, mod_count, ):
    gene_space, gene_type = generate_gene(parameter_base, mod_connections, mod_count)
    genome = []

    for gene,type in zip(gene_space, gene_type):
        if type is float:
            value = np.random.uniform(gene["low"], gene["high"])
        if type is int:
            value = np.random.randint(gene["low"], gene["high"]+1)
        genome.append(value)

    return genome
        

def create_random_patch(synth,name,parameter_base, modulations):
    path = "patches/" + name + ".vital"

    genome = create_random_genome(parameter_base, modulations, MOD_COUNT)
    set_synth(synth,genome, parameter_base, modulations, ENV_COUNT, LFO_COUNT)

    make_vital(synth,path)


def generate_gene(parameter_base, mod_connections, mod_count):
    connection_count = len(mod_connections)
    gene_space = []
    for p in parameter_base:
        gene_space.append({'low': p.min, 'high': p.max})
    for i in range (mod_count):
        gene_space.append({'low': 0, 'high': (connection_count-1)})

    gene_type = (
        [float] * len(parameter_base) +
        [int] * mod_count
    )
    return gene_space, gene_type







#=============================================================================+
#________________________PARAMETER SETUP FUNCTIONS____________________________
#=============================================================================+

def add_p(controls, PARAMETER_BASE, i, parameter):
    name,min,max = parameter[0],parameter[1],parameter[2]
    name = name.replace("xx",str(i))
    default_value = controls[name].get_normalized()
    if (max > 1):
        default_value = controls[name].value()
    p = Parameter(name,default_value,min,max)
    PARAMETER_BASE.append(p)

def build_parameter_base(parameter_lists):
    parameter_base = []

    for (content,count) in (parameter_lists):
        add_parameters(parameter_base, content, count)

    return parameter_base


    


def add_parameters(PARAMETER_BASE, parameters, count):
    synth = vita.Synth()
    controls = synth.get_controls()
    for i in range (1, count +1):
        for p in parameters:
            add_p(controls, PARAMETER_BASE, i,p)



#10x * 100x


#=============================================================================+
#__________________________VITAL SYNTH FUNCTIONS_____________________________
#=============================================================================+

def set_synth(synth, chromosome, parameter_base, mod_list, env_count, lfo_count):
    controls = synth.get_controls()
    synth.load_json(DEFAULT_STATE)
    synth.clear_modulations()
    assert (len(chromosome) == len(parameter_base)+ env_count+lfo_count)

    for i in range (env_count):
        j = len(parameter_base)+i
        connection_target = mod_list[chromosome[j]]
        assert synth.connect_modulation("env_"+str(i+2), connection_target)

    for i in range (lfo_count):
        j = len(parameter_base)+env_count+i
        connection_target = mod_list[chromosome[j]]
        assert synth.connect_modulation("lfo_"+str(i+1), connection_target)
        controls["lfo_"+str(i+1)+"_sync"].set(0)

    for i in range (len(parameter_base)):
        p = parameter_base[i]
        if (p.max > 1):
            N = p.max
            #x = min(int(p.value * (N+1)),N)
            controls[p.name].set(chromosome[i])
        else:
            controls[p.name].set_normalized(chromosome[i])



def render_synth(synth):
    audio = synth.render(PITCH, VELOCITY, NOTE_DUR, RENDER_DUR)
    if (audio.ndim == 2):
        audio = np.mean(audio, axis=0)
    audio = audio.astype(np.float32)
    return audio


def make_vital(synth, preset_path):
    json_text = synth.to_json()
    with open(preset_path, "w") as f:
        f.write(json_text)
    with open(preset_path, "r") as f:
        json_text = f.read()
    change_version(preset_path)


#=============================================================================+
#___________________________FITNESS CALC FUNCTIONS____________________________
#=============================================================================+

def target_similarity(A,metric,jtfs_x=None):
    match metric:
        case "SpE":
            A = create_mel_spec(A)
            return spectral_error(A,TARGET_SPEC)
        case "mfcc":
            A = mfcc_mean(A)
            return mfcc_distance(A, TARGET_MFCC_MEAN)
        case "jtfs":
            A = extract_jtfs(A,worker_jtfs)
            return jtfs_distance(A,TARGET_JTFS)
        case "dtw":
            A = dtw_prep(A, TARGET_REF_MEAN, TARGET_REF_STD)
            return dtw_distance(A, TARGET_DTW)
        case "SpC":
            A = spectral_centroid(A)
            return centroid_distance(A, TARGET_CENTROID)
    raise Exception("!!!! METRIC " + metric + " NOT FOUND !!!!")

       
def mfcc_distance(mfccA, mfccB):
    similarity = 1 - cosine(mfccA, mfccB)
    return similarity

def spectral_error(specA, specB):

    score = 1 - (abs(specA-specB).mean() /max(specA.mean(), specB.mean()))
    return score


def jtfs_distance(jtfsA, jtfsB):
    distance = np.linalg.norm(jtfsA - jtfsB)
    score = np.exp(-0.1 * distance)
    return score

def extract_jtfs(audio, jtfs_x):
    #audio = audio/np.sqrt(np.mean(audio**2) +1e-12)

    S = jtfs_x(torch.tensor(audio).float())
    S = S.numpy()
    #features = S.mean(axis=-1)

    return S.ravel()

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

def dtw_prep(audio, ref_mean=None, ref_std=None):
    mfcc = extract_mfcc(audio).T[:, 1:]
    if (ref_mean is None):
        ref_mean = mfcc.mean(axis=0, keepdims=True)
        ref_std = mfcc.std(axis=0, keepdims=True)
        ref_std[ref_std < 1e-8] = 1e-8
        a = np.ascontiguousarray((mfcc - ref_mean) / ref_std, dtype=np.float64)
        return a, ref_mean, ref_std
    a = np.ascontiguousarray((mfcc - ref_mean) / ref_std, dtype=np.float64)
    return a

def dtw_distance(A, B):
    raw_distance = dtw_ndim.distance_fast(A,B)
    distance = raw_distance /(len(A) + len(B))
    similarity = 1.0 / (1.0 + distance)

    return similarity

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

def mfcc_mean(audio):
    
    mfcc = extract_mfcc(audio)
    mean = np.mean(mfcc, axis=1)
    return mean

def extract_mfcc(audio):
    mfcc = librosa.feature.mfcc(
            y=audio,
            sr=SAMPLE_RATE,
            n_mfcc=MFCC_COUNT
        )
    return mfcc


#=============================================================================+
#______________________________PYGAD FUNCTIONS________________________________
#=============================================================================+

def fitness_func(ga_instance, solution, solution_i):
    #print("fitnessing")
    #print(f"Fitness PID={os.getpid()} solution={solution_i}", flush=True)
    global worker_synth
    if worker_synth is None:
        worker_synth = vita.Synth()
        #print(f"Created synth in PID {os.getpid()}")

   

    set_synth(worker_synth,solution, PARAMETER_BASE, MOD_CONNECTIONS, ENV_COUNT, LFO_COUNT)
    attempt_audio = render_synth(worker_synth)

    if (current_metric == "jtfs"):
        global worker_jtfs
        if worker_jtfs is None:
            worker_jtfs = TimeFrequencyScattering(
                J=8,
                J_fr=2,
                shape=len(TARGET_AUDIO),
                Q=(8,1)
                )
        score = target_similarity(attempt_audio, current_metric, worker_jtfs)
    else:
        score = target_similarity(attempt_audio, current_metric)


    return score

def on_parents(ga_instance, selected_parents):
    immigrant_count = 4*SCALE
    indices = np.random.choice(selected_parents.shape[0],size=immigrant_count, replace=False)
    parameter_base = PARAMETER_BASE
    mods = MOD_CONNECTIONS
    m_count = MOD_COUNT

    for i in indices:
        selected_parents[i]=create_random_genome(parameter_base,mods,m_count)

def on_generation(ga_instance):
    update_bar()

    solution, fitness, solution_idx = ga_instance.best_solution()
    i = ga_instance.generations_completed
    set_synth(main_synth,solution, PARAMETER_BASE, MOD_CONNECTIONS, ENV_COUNT, LFO_COUNT)
    make_vital(main_synth, "candidates/candidate_%d.vital" %i)





















#=============================================================================+
#____________________________________MAIN_____________________________________
#=============================================================================+


metric = sys.argv[1]

metrics = [metric]
targets_n = sys.argv[2:]
targets = []
for t in targets_n:
    targets.append("target_"+t)

MFCC_COUNT = 20 
SAMPLE_RATE = 44_100
BPM = 120.0
NOTE_DUR = 2#1.2
RENDER_DUR = 3#1.41
PITCH = 48
VELOCITY = 0.7

OSC_COUNT = 2       #max: 3
FILTER_COUNT = 2    #max: 2
ENV_COUNT = 3      #max: 5 EXCLUDING MAIN VOLUME ENVELOPE
LFO_COUNT = 2
MOD_COUNT = ENV_COUNT + LFO_COUNT





main_synth = vita.Synth()
controls = main_synth.get_controls()

main_synth.set_bpm
main_synth.load_preset("default.vital")
DEFAULT_STATE = main_synth.to_json()

worker_synth = None

FILTER_STYLES = [
    '12dB', 
    '24dB', 
    'Notch Blend', 
    'Notch Spread', 
    'B/P/N', 
    'Sin', 
    'Saturated Sin', 
    'Triangle', 
    'Square', 
    'Pulse']
MOD_CONNECTIONS = [
    "filter_1_cutoff",
    "filter_2_cutoff",
    "osc_2_level",
    #"osc_3_level",
    #"sample_level",
    "osc_1_wave_frame",
    "osc_2_wave_frame",
    #"osc_3_wave_frame"
    #"osc_2_spectral_morph_amount",
    #"osc_3_spectral_morph_amount"
]
#parameter name, min value, max value
osc_parameters = [
    ["osc_xx_unison_detune", 0.0, 0.4],
    ["osc_xx_unison_voices", 0.0, 0.2],
    ["osc_xx_wave_frame", 0.0, 1.0],
    #["osc_xx_spectral_morph_type", 0.0, 12],
    #["osc_xx_spectral_morph_amount", 0.0, 1],
    ["osc_xx_level", 0.0, 1.0]
]
filter_parameters = [
    ["filter_xx_mix", 0, 1.0],
    ["filter_xx_cutoff", 0, 1.0],
    ["filter_xx_style", 0, 4.0],
    ["filter_xx_resonance", 0.0, 1.0],
    ["filter_xx_blend", 0.0, 1.0],
    ["filter_xx_drive", 0.0, 1.0]
]
env_parameters = [
    ["env_xx_attack", 0.0, 0.5],
    ["env_xx_decay", 0.1, 0.5],
    ["env_xx_sustain", 0.0, 1.0],
    ["env_xx_release", 0.0, 0.5],
    ["modulation_xx_amount", 0.0, 1.0]
]

lfo_parameters = [
    ["lfo_xx_frequency", 0.0, 1.0]
]


selection = [
    (osc_parameters, OSC_COUNT),
    (filter_parameters, FILTER_COUNT),
    (env_parameters, ENV_COUNT+1),
    (lfo_parameters, LFO_COUNT)
]

PARAMETER_BASE = build_parameter_base(selection)



#Target 1:  LFO filter, env filter
#Target 2:  2nd oscillator fade
#Target 3:  LFO vol
#Target 4:  attack, env filter, unison



#10x*100x = 1000x²

SCALE = 1

gene_space, gene_type = generate_gene(PARAMETER_BASE, MOD_CONNECTIONS, MOD_COUNT)
num_generations = SCALE*10
sol_per_pop = SCALE*100
num_parents_mating = SCALE*80#940
num_genes = len(PARAMETER_BASE)+MOD_COUNT
save_best_solutions=False
random_seed=12345678
mutation_type="random"
mutation_probability = 1/len(gene_space)+0.1
parent_selection_type="tournament"
K_tournament= SCALE*1
elitism=SCALE*1
crossover_type="uniform"
crossover_probability=1
processes=["process", 3]


for metric in metrics:
    for target in targets:
        gene_space, gene_type = generate_gene(PARAMETER_BASE, MOD_CONNECTIONS, MOD_COUNT)


        current_target = target
    
        current_metric = metric

        #create_random_patch(main_synth,"random_target",PARAMETER_BASE,MOD_CONNECTIONS)
        print(current_target)
        main_synth.load_preset("targets/"+current_target+".vital")

        TARGET_AUDIO = render_synth(main_synth)
        TARGET_SPEC = create_mel_spec(TARGET_AUDIO)
        TARGET_MFCC_MEAN = mfcc_mean(TARGET_AUDIO)
        TARGET_MFCC = extract_mfcc(TARGET_AUDIO)
        TARGET_DTW, TARGET_REF_MEAN, TARGET_REF_STD,= dtw_prep(TARGET_AUDIO)
        TARGET_CENTROID = spectral_centroid(TARGET_AUDIO)

        main_jtfs = TimeFrequencyScattering(
            J=8,
            J_fr=2,
            shape=len(TARGET_AUDIO),
            Q=(8,1)
        )
        worker_jtfs = None
        TARGET_JTFS = extract_jtfs(TARGET_AUDIO,main_jtfs)
        graph_spec(TARGET_SPEC, current_target)


        wavfile.write("audio/"+current_target,SAMPLE_RATE,TARGET_AUDIO.T)
        
        print("Start time: ",datetime.now().strftime("%H:%M:%S"))
        pbar = tqdm(
            total=num_generations,
            desc="Progress",
            ncols=100
        )
        start = time.time()

        ga_instance = pygad.GA(num_generations=num_generations,
                            #parallel_processing=processes,
                            num_parents_mating=num_parents_mating,
                            fitness_func=fitness_func,
                            sol_per_pop=sol_per_pop,
                            num_genes=num_genes,
                            save_best_solutions=save_best_solutions,
                            #random_seed=random_seed,
                            gene_space=gene_space,
                            gene_type=gene_type,
                            mutation_type=mutation_type,
                            mutation_probability=mutation_probability,
                            on_generation=on_generation,
                            on_parents=on_parents,
                            #on_fitness=on_fitness,
                            parent_selection_type=parent_selection_type,
                            K_tournament=K_tournament,
                            keep_elitism=elitism,
                            crossover_type=crossover_type,
                            crossover_probability=crossover_probability
                            )
        ga_instance.run()
    
        solution, solution_fitness, solution_idx = ga_instance.best_solution()

        t = time.time() - start
        print ("duration:", format_time(t))
        print("fitness: ", solution_fitness)


        
        
        set_synth(main_synth, solution, PARAMETER_BASE, MOD_CONNECTIONS, ENV_COUNT, LFO_COUNT)
        final_audio = render_synth(main_synth)

        score_str = str(round(solution_fitness, 5))
        filename = metric+"_on_"+target+"="+score_str
        
        make_vital(main_synth, "patches/"+filename+".vital")
        final_spec = create_mel_spec(final_audio)
        graph_spec(final_spec, filename)

        wavfile.write("audio/"+filename+".wav", SAMPLE_RATE, final_audio.T)

        fitnessfilename = "generation_log/" +filename + "_fitnesses.txt"
        with open(fitnessfilename, "w") as f:
            #f.write("Generation\tBest Fitness\n")
            for generation, fitness in enumerate(ga_instance.best_solutions_fitness):
                f.write(f"{generation}\t{fitness}\n")


        #ga_instance.plot_fitness()








