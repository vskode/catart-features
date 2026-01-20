import bacpipe
import os
from pathlib import Path
import json

import pandas as pd
import numpy as np
import json

from audio_prep import get_acoustic_indices

data_dir = Path('data/AnuraMiniSet')

def get_bacpipe_features():
    bacpipe.config.audio_dir = data_dir
    bacpipe.config.dashboard = False
    bacpipe.settings.run_default_classifier = False
    bacpipe.play()


    umap_embeddings = {}
    for model in os.listdir(f'bacpipe_results/{data_dir.stem}/evaluations'):
        file_path = list(Path(f'bacpipe_results/{data_dir.stem}/dim_reduced_embeddings').rglob(f'*{model}*'))[0]
        with open(file_path / f'{data_dir.stem}_umap.json', 'r') as f:
            umap_embeddings[model] = json.load(f)
    return umap_embeddings

def get_umap_2d(embeds):
    df = pd.DataFrame()
    x, y = {}, {}
    for model, embed in embeds.items():
        x[model] = embed['x']
        y[model] = embed['y']

    first_model = list(embeds.values())[0]
    file_names = []
    [file_names.extend([f] * ll) for f, ll in zip(first_model['metadata']['audio_files'], 
                                                  first_model['metadata']['nr_embeds_per_file'])]


    starts = []
    # was previously 1000 
    [starts.extend((np.arange(ll)*3000).tolist()) for ll in first_model['metadata']['nr_embeds_per_file']]

    duration = [int((first_model['metadata']['segment_length (samples)'] / first_model['metadata']['sample_rate (Hz)']) * 1000)] * len(list(x.values())[0])

    df['file_names'] = file_names
    df['starts'] = starts
    df['duration'] = duration
    
    for model in embeds.keys():
        df[f'{model}1'] = x[model]
        df[f'{model}2'] = y[model]

    return df

def concatenate_features(df):
    for file in df.file_names.unique():
        df_temp = df[df.file_names == file]
        df_temp.to_csv(
            f"/media/lorenzo-dubois/TRIM_4/files/{file.split('/')[-1].replace('wav', 'txt')}", 
            sep=' ', 
            index=False,
            header=None
            )

bacpipe_features = get_bacpipe_features()
# indices = get_acoustic_indices()

# TODO:vincent fix different input lengths with padding
        
df = get_umap_2d(bacpipe_features)

concatenate_features(df)