import bacpipe
import os
from pathlib import Path
import json

import pandas as pd
import numpy as np
import json

from audio_prep import get_acoustic_indices, CATART_AUDIO_LENGTH, SELECTED_COLUMNS
from librosa import get_duration

def get_bacpipe_features(DATA_DIR, models=None):
    # bacpipe.settings.main_results_dir = Path('G:\Work\Embeddings')
    bacpipe.settings.main_results_dir = Path('/mnt/swap/Work/Embeddings/catart')
    bacpipe.config.audio_dir = Path(DATA_DIR)
    if models:
        bacpipe.config.models = models
    else:
        bacpipe.config.already_computed = True
    bacpipe.config.dashboard = False
    bacpipe.settings.run_pretrained_classifier = False
    bacpipe.settings.device = 'cuda'
    
    bacpipe.settings.only_embed_annotations = True
    
    df = make_annotations_for_bacpipe_inputs()
    
    bacpipe.play()

    ld = bacpipe.model_specific_embedding_creation(
        **vars(bacpipe.config), **vars(bacpipe.settings)
    )


    umap_embeddings = {}
    for model in ld.keys():
        file_path = list(ld[model].paths.dim_reduc_parent_dir.rglob(f'*{model}'))[0]
        with open(list(file_path.rglob('*.json'))[0], 'r') as f:
            umap_embeddings[model] = json.load(f)
            
    if len(df) != ld[model].metadata_dict['nr_embeds_total']:
        ld[model].audio_dir = DATA_DIR
        make_annotations_for_bacpipe_inputs(ld[model])
    
    
    return umap_embeddings, ld[model]


def make_annotations_for_bacpipe_inputs(loader=False):
    """
    Build an annotations.csv file which can then be used as an input
    for bacpipe. This way all deep learning models create embeddings
    based on the same sounds even if they require different input
    lengths. If the audio length are shorter than the model-specific
    input length, the audio is minimum padded to correspond to the
    required model input length. 
    """
    if not loader:
        configs = {**vars(bacpipe.config)}
        configs.pop('dim_reduction_model')
        loader = bacpipe.generate_embeddings.Loader(
            model_name = 'birdnet', 
            check_if_combination_exists=False,
            dim_reduction_model=None, 
            **configs, 
            **vars(bacpipe.settings)
            )
        
        # specify_annotation_grid 
        lengths = [get_duration(path=f) for f in loader.files]
        segments_per_file = [
            # we decided to discard the last segment that is under
            # CATART_AUDIO_LENGTH seconds long
            l // CATART_AUDIO_LENGTH + 1
            for l in lengths
        ]
        file_array_same_length_as_starts = []
        [
            file_array_same_length_as_starts.extend(
                [str(file.relative_to(loader.audio_dir))] * int(nr_segs)
            )
            for file, nr_segs in zip(
                loader.files, segments_per_file
                )
        ]
    else:
        segments_per_file = loader.metadata_dict['files']['nr_embeds_per_file']
        file_array_same_length_as_starts = []
        [
            file_array_same_length_as_starts.extend(
                [file] * int(nr_segs)
            )
            for file, nr_segs in zip(
                loader.metadata_dict['files']['audio_files'], segments_per_file
                )
        ]
        
    starts = []
    [
        starts.extend(
            np.arange(nr_segs)
            *CATART_AUDIO_LENGTH
            ) 
        for nr_segs in segments_per_file
        ]
    
    catart_grid = pd.DataFrame()
    catart_grid['start'] = starts
    catart_grid['end'] = catart_grid['start'] + CATART_AUDIO_LENGTH
    catart_grid['audiofilename'] = file_array_same_length_as_starts
    catart_grid['label:speices'] = [None] * len(catart_grid)
    catart_grid.to_csv(Path(loader.audio_dir) / 'annotations.csv')
    return catart_grid

def get_umap_2d(data_dir, embeds):
    df = pd.DataFrame()

    annotations = pd.read_csv(
        Path(data_dir) / 'annotations.csv'
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
    