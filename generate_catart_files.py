import bacpipe
import os
from pathlib import Path
import json

import pandas as pd
import numpy as np
import json

from audio_prep import get_acoustic_indices, CATART_AUDIO_LENGTH, SELECTED_COLUMNS
from librosa import get_duration

def get_bacpipe_features(MODELS, DATA_DIR):
    bacpipe.settings.main_results_dir = Path('G:\Work\Embeddings')
    bacpipe.config.audio_dir = Path(DATA_DIR)
    bacpipe.config.models = MODELS
    bacpipe.config.dashboard = False
    bacpipe.settings.run_pretrained_classifier = False
    bacpipe.settings.device = 'cpu'
    
    bacpipe.settings.only_embed_annotations = True
    
    make_annotations_for_bacpipe_inputs()
    
    bacpipe.play()

    ld = bacpipe.model_specific_embedding_creation(
        **vars(bacpipe.config), **vars(bacpipe.settings)
    )

    umap_embeddings = {}
    for model in ld.keys():
        file_path = list(ld[model].paths.dim_reduc_parent_dir.rglob(f'*{model}'))[0]
        with open(list(file_path.rglob('*.json'))[0], 'r') as f:
            umap_embeddings[model] = json.load(f)
            
    audio_file_paths = [Path(DATA_DIR) / f for f in ld[model].metadata_dict['files']['audio_files']]
    return umap_embeddings, audio_file_paths


def make_annotations_for_bacpipe_inputs():
    """
    Build an annotations.csv file which can then be used as an input
    for bacpipe. This way all deep learning models create embeddings
    based on the same sounds even if they require different input
    lengths. If the audio length are shorter than the model-specific
    input length, the audio is minimum padded to correspond to the
    required model input length. 
    """
    configs = {**vars(bacpipe.config)}
    configs.pop('dim_reduction_model')
    ld = bacpipe.generate_embeddings.Loader(
        model_name = 'birdnet', 
        check_if_combination_exists=False,
        dim_reduction_model=None, 
        **configs, 
        **vars(bacpipe.settings)
        )
    
    # specify_annotation_grid 
    lengths = [get_duration(path=f) for f in ld.files]
    segments_per_file = [
        # we decided to discard the last segment that is under
        # CATART_AUDIO_LENGTH seconds long
        l // CATART_AUDIO_LENGTH + 1
        for l in lengths
    ]
    starts = []
    [
        starts.extend(
            np.arange(nr_segs)
            *CATART_AUDIO_LENGTH
            ) 
        for nr_segs in segments_per_file
        ]
    file_array_same_length_as_starts = []
    [
        file_array_same_length_as_starts.extend(
            [str(file.relative_to(ld.audio_dir))] * int(nr_segs)
        )
        for file, nr_segs in zip(ld.files, segments_per_file)
    ]
    
    catart_grid = pd.DataFrame()
    catart_grid['start'] = starts
    catart_grid['end'] = catart_grid['start'] + CATART_AUDIO_LENGTH
    catart_grid['audiofilename'] = file_array_same_length_as_starts
    catart_grid['label:speices'] = [None] * len(catart_grid)
    catart_grid.to_csv(ld.audio_dir / 'catart_timestamp_annotations.csv')

def get_umap_2d(data_dir, embeds):
    df = pd.DataFrame()
    x, y = {}, {}
    for model, embed in embeds.items():
        x[model] = embed['x']
        y[model] = embed['y']

    annotations = pd.read_csv(
        Path(data_dir) / 'catart_timestamp_annotations.csv'
        )

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
    df = pd.concat([df_bacpipe, df_indices], axis=1)
    df.to_csv(Path(data_dir) / 'catart_features.txt', index=False, sep=' ')
