from generate_catart_files import get_bacpipe_features, get_umap_2d, concatenate_features
from audio_prep import get_acoustic_indices
from pathlib import Path
import pandas as pd

# global config
MODELS = ['birdnet']#, 'perch_bird', 'beats', 'naturebeats']
# DATA_DIR = '/media/siriussound/Extreme SSD/Recordings/terrestrial/Birds/Lorenzo/Bois_Lavigne_04_2024'
DATA_DIR = r'E:\Recordings\terrestrial\Amphibians\AnuranSet\AnuranSet'

bacpipe_features, audio_file_paths = get_bacpipe_features(MODELS, DATA_DIR)

if (Path(DATA_DIR) / 'acoustic_indices.csv').exists():
    df_indices = pd.read_csv(Path(DATA_DIR) / 'acoustic_indices.csv')
    result_dict = {col: df_indices[col].values for col in df_indices.columns}
    indices = result_dict
else:
    indices = get_acoustic_indices(audio_file_paths)
        
df_features = get_umap_2d(DATA_DIR, bacpipe_features)

concatenate_features(DATA_DIR, df_features, indices)
