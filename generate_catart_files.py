import bacpipe
import os
from pathlib import Path
import json

import pandas as pd
import numpy as np
import json

from audio_prep import get_acoustic_indices, SELECTED_COLUMNS
from librosa import get_duration

def get_bacpipe_features(audio_dir, model_for_time_grid, models=None, device='cuda'):
    
    df = make_annotations_for_bacpipe_inputs(audio_dir, model_for_time_grid)
    
    loader = dict()
    umaps = {}
    for model in models:
        
        loader = bacpipe.generate_embeddings(
            model_name=model, 
            audio_dir=audio_dir, 
            annotations_df=df,
            dim_reduction_model=None,
            device=device, 
            only_embed_annotations=True, 
            annotations_filename='catart_grid.csv',
            run_pretrained_classifier=False
            )
        bacpipe.generate_embeddings(
            model_name=model, 
            audio_dir=audio_dir, 
            annotations_df=df,
            dim_reduction_model='umap',
            device=device, 
            only_embed_annotations=True, 
            annotations_filename='catart_grid.csv',
            run_pretrained_classifier=False
            )
    
        file_path = list(loader.paths.dim_reduc_parent_dir.rglob(f'*{model}'))[0]
        with open(list(file_path.rglob('*.json'))[0], 'r') as f:
            umaps[model] = json.load(f)
    
    return umaps, audio_dir, loader


def make_annotations_for_bacpipe_inputs(audio_dir, model_for_time_grid):
    """
    Build an annotations.csv file which can then be used as an input
    for bacpipe. This way all deep learning models create embeddings
    based on the same sounds even if they require different input
    lengths. If the audio length are shorter than the model-specific
    input length, the audio is minimum padded to correspond to the
    required model input length. 
    """
    df_ml = bacpipe.metadata_labels(audio_dir, model=model_for_time_grid, as_type='dataframe', overwrite=False)
    
    catart_grid = df_ml[['start', 'end', 'audio_file_name']].copy()
    catart_grid = catart_grid.rename(columns={'audio_file_name': 'audiofilename'})
    catart_grid['audiofilename'] = [Path(s).relative_to(audio_dir) for s in catart_grid['audiofilename'].values]
    catart_grid.loc[:, 'label:speices'] = [None] * len(catart_grid)
    catart_grid.to_csv(Path(audio_dir) / 'catart_grid.csv')
    return catart_grid

def get_umap_2d(data_dir, embeds):
    df = pd.DataFrame()

    annotations = pd.read_csv(
        Path(data_dir) / 'catart_grid.csv'
        )
    annotations, embeds = ensure_common_files(embeds, annotations)
    
    x, y = {}, {}
    for model, embed in embeds.items():
        x[model] = embed['x']
        y[model] = embed['y']

    duration = annotations['end'] - annotations['start']

    df['Filename'] = [str(Path(f).as_posix()) for f in annotations['audiofilename']]
    df['start'] = annotations['start'].astype(int) * 1000
    df['Duration'] = duration.astype(int) * 1000
    
    for model in embeds.keys():
        if not len(x[model]) == len(annotations):
            print("length of embeddings and annotations don't match")
            x[model] = x[model][:len(annotations)]
            y[model] = y[model][:len(annotations)]
            
        df[f'{model}1'] = x[model]
        df[f'{model}2'] = y[model]

    return df

def concatenate_features(data_dir, df_bacpipe, indices):
    df_indices = pd.DataFrame({k: v for k, v in indices.items() if k in SELECTED_COLUMNS})
    df_indices['index'] = df_bacpipe.index
    df_indices = df_indices.set_index('index')
    df = pd.concat([df_bacpipe, df_indices], axis=1)
    df['Filename'] = [filename.split('/')[-1] for filename in df['Filename']]
    df.to_csv(Path(data_dir) / 'catart_features.txt', index=False, sep=' ')
    df.to_csv('catart_features.txt', index=False, sep=' ')

def ensure_common_files(embeds, annotations):
    shared_files = []
    for model in embeds.keys():
        embed_files = np.unique(embeds[model]['metadata']['audio_files']).tolist()
        annot_files = annotations['audiofilename'].unique().tolist()
        intersect = list(set(embed_files).intersection(annot_files))
        if len(shared_files) == 0:
            shared_files = intersect
        else:
            shared_files = list(set(shared_files).intersection(intersect))
        
        bool_filter_embeds = []
        [
            bool_filter_embeds.extend([file in shared_files] * n) for n, file in 
            zip(embeds[model]['metadata']['nr_embeds_per_file'], embeds[model]['metadata']['audio_files'])
        ]
        for k, v in embeds[model].items():
            if k in ['x', 'y', 'timestamp']:
                embeds[model][k] = np.array(v)[bool_filter_embeds].tolist()
        
        annotations = annotations[annotations['audiofilename'].isin(shared_files)]
        
        #### ensure correct mappint by comparing timestamps
        results = len(annotations['start']) == len(embeds[model]['timestamp'])
        if results:
            print('Mapping was successful, all timestamps and files match')
        else:
            print("Mapping was unsuccessful, timestamps and files don't match")
    return annotations, embeds
    