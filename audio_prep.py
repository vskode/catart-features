import pandas as pd
import os
import time
import numpy as np
from tqdm import tqdm                          
from concurrent import futures                 
from maad import sound, features
import multiprocessing as mp
import platform
import librosa as lb
import warnings
from pathlib import Path

CATART_AUDIO_LENGTH = 3

# Select only ecoacoustic indices needed for Catart : scikit-maad.github.io/
SELECTED_COLUMNS = [
    'ZCR', 'MFC',
    'LFC', 'AGI', 'nROI', 'LEQt', 'LEQf', 
    'SNRf', 'SNRt', 'BGNf', 'BGNt', 'HFC'
]

def single_file_processing(audio_path, sr, nr_embeds_per_file):

    S = -35         # sensitivity
    G = 26 + 16     # gain    

    try:

        with warnings.catch_warnings(action="ignore"):
            wave, fs = lb.load(
                audio_path,
                mono=False,
                sr=sr
            )

            
            total_samples = len(wave)
            n_windows = nr_embeds_per_file

            rows = []

            for i in range(n_windows):
                start_s = i * CATART_AUDIO_LENGTH
                start_sample = int(start_s * fs)
                end_sample = int(min((start_s + CATART_AUDIO_LENGTH) * fs, total_samples))

                if start_sample >= total_samples:
                    break

                segment = wave[start_sample:end_sample]
                if len(segment) == 0:
                    continue

                # === Temporal indices ===
                df_audio_ind = features.all_temporal_alpha_indices(
                    segment, fs,
                    gain=G,
                    sensibility=S,
                    dB_threshold=3,
                    rejectDuration=0.01,
                    verbose=False
                )

                # === Spectral indices ===
                Sxx_power, tn, fn, ext = sound.spectrogram(
                    segment, fs,
                    window='hann',
                    nperseg=1024,
                    noverlap=1024 // 2,
                    verbose=False
                )

                df_spec_ind, _ = features.all_spectral_alpha_indices(
                    Sxx_power, tn, fn,
                    flim_low=[0, 1500],
                    flim_mid=[1500, 8000],
                    flim_hi=[8000, 20000],
                    gain=G,
                    sensitivity=S,
                    verbose=False
                )
                if False: #DEBUG
                    from matplotlib import pyplot as plt
                    fig, ax = plt.subplots()
                    S_dB = lb.power_to_db(Sxx_power, ref=np.max)
                    img = lb.display.specshow(S_dB, x_axis='time',
                                            y_axis='mel', sr=sr,
                                            fmax=8000, ax=ax)
                    fig.colorbar(img, ax=ax, format='%+2.0f dB')
                    ax.set(title='Mel-frequency spectrogram')
                    fig.savefig('test.png')

                # >>> MODIFIED (same merge strategy as GitHub)
                df_row = pd.concat([df_audio_ind, df_spec_ind], axis=1)
                df_row.insert(0, 'file', audio_path)
                df_row.insert(1, 'start (ms)', start_s * 1000)

                rows.append(df_row)

            return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return pd.DataFrame()


def get_acoustic_indices(DATA_DIR, loader):
    
    audio_files = [
        Path(DATA_DIR) 
        / f for f in loader.metadata_dict['files']['audio_files']
        ]
    sr = loader.metadata_dict['sample_rate (Hz)']
    nr_embeds_per_file = loader.metadata_dict['files']['nr_embeds_per_file']
    
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method("fork")

    
    df_indices = pd.DataFrame()

    tic = time.perf_counter()
    

    if platform.system() == 'Linux': # multiprocessing works on linux
        nb_cpu = os.cpu_count() - 2
        with tqdm(total=len(audio_files), desc="multi cpu indices calculation...") as pbar:
            with futures.ProcessPoolExecutor(max_workers=nb_cpu) as pool:
                for df_tmp in pool.map(
                    single_file_processing, 
                    audio_files, 
                    [sr]*len(audio_files),
                    nr_embeds_per_file
                    ):
                    df_indices = pd.concat([df_indices, df_tmp])
                    pbar.update(1)
    else: # it does not on windows
        with tqdm(total=len(audio_files), desc="processing files with single process") as pbar:
            for file in audio_files:
                df_tmp = single_file_processing(file, sr, nr_embeds_per_file)
                df_indices = pd.concat([df_indices, df_tmp])
                pbar.update(1)

    toc = time.perf_counter()
    print(f"Elapsed time (multi CPU): {toc - tic:.1f} s")
    
    # result_dict = {col: df_indices[col].values for col in df_indices.columns}
    # df_indices = result_dict
    return df_indices


# def get_acoustic_indices(audio_files):
#     # Remove the set_start_method line from here
#     nb_cpu = os.cpu_count() - 2
    
#     # Use a list to collect DataFrames (MUCH faster than pd.concat in a loop)
#     all_dfs = []

#     with tqdm(total=len(audio_files), desc="multi cpu indices calculation...") as pbar:
#         with futures.ProcessPoolExecutor(max_workers=nb_cpu) as pool:
#             # map returns an iterator, we collect it
#             for df_tmp in pool.map(single_file_processing, audio_files):
#                 if not df_tmp.empty:
#                     all_dfs.append(df_tmp)
#                 pbar.update(1)

#     if all_dfs:
#         df_indices = pd.concat(all_dfs, ignore_index=True)
#     else:
#         df_indices = pd.DataFrame()

#     # Convert to dict as per your requirements
#     return {col: df_indices[col].values for col in df_indices.columns}