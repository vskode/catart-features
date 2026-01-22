import pandas as pd
import os
import time
import numpy as np
from tqdm import tqdm                          
from concurrent import futures                 
from maad import sound, features
import multiprocessing as mp

def single_file_processing(audio_path, window_s=3):

    S = -35         # sensitivity
    G = 26 + 16     # gain    

    try:
        # Load audio
        wave, fs = sound.load(
            filename=audio_path,
            channel='left',
            detrend=True,
            verbose=False
        )

        
        total_samples = len(wave)
        duration_s = total_samples / fs
        n_windows = int(np.ceil(duration_s / window_s)) # >>>>> ceil ? claude

        rows = []

        for i in range(n_windows):
            start_s = i * window_s
            start_sample = int(start_s * fs)
            end_sample = int(min((start_s + window_s) * fs, total_samples))

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

            # >>> MODIFIED (same merge strategy as GitHub)
            df_row = pd.concat([df_audio_ind, df_spec_ind], axis=1)
            df_row.insert(0, 'file', audio_path)
            df_row.insert(1, 'start (ms)', start_s * 1000)

            rows.append(df_row)

        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return pd.DataFrame()


def get_acoustic_indices(path):
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method("fork")

    audio_files = [
        os.path.join(path, f)
        for f in os.listdir(path)
        if f.endswith(".wav")
    ]
    
    nb_cpu = os.cpu_count() - 2
    df_indices = pd.DataFrame()

    tic = time.perf_counter()

    with tqdm(total=len(audio_files), desc="multi cpu indices calculation...") as pbar:
        with futures.ProcessPoolExecutor(max_workers=nb_cpu) as pool:
            for df_tmp in pool.map(single_file_processing, audio_files):
                df_indices = pd.concat([df_indices, df_tmp])
                pbar.update(1)

    toc = time.perf_counter()
    print(f"Elapsed time (multi CPU): {toc - tic:.1f} s")

    # Select only ecoacoustic indices needed for Catart : scikit-maad.github.io/
    selected_columns = [
        'file', 'start (ms)', 'ZCR', 'MFC',
        'LFC', 'AGI', 'nROI', 'LEQt', 'LEQf', 
        'SNRf', 'SNRt', 'BGNf', 'BGNt', 'HFC'
    ]
    
    df_indices = df_indices[selected_columns]
    result_dict = {col: df_indices[col].values for col in df_indices.columns}
    df_indices = result_dict
    return df_indices
