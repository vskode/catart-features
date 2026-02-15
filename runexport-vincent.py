from generate_catart_files import get_bacpipe_features, get_umap_2d, concatenate_features
from audio_prep import get_acoustic_indices
from pathlib import Path
import pandas as pd

# global config
MODELS = ['birdnet', 'beats', 'naturebeats']
# DATA_DIR = '/media/siriussound/Extreme SSD/Recordings/terrestrial/Birds/BirdSet/HSN - soundscapes high Sierra Nevada'
# DATA_DIR = '/media/siriussound/Extreme SSD/Recordings/terrestrial/Birds/Lorenzo/Bois_Lavigne_04_2024'
# DATA_DIR = '/media/siriussound/Extreme SSD/Recordings/terrestrial/Amphibians/AnuranSet/AnuranSet'
DATA_DIR = '/media/siriussound/Extreme SSD/Recordings/MyRecordings/AudioMoths_roadtrip'
# DATA_DIR = r'E:\Recordings\terrestrial\Amphibians\AnuranSet\AnuranSet'

bacpipe_features, loader = get_bacpipe_features(DATA_DIR, MODELS)

if (Path(DATA_DIR) / 'acoustic_indices.csv').exists():
    df_indices = pd.read_csv(Path(DATA_DIR) / 'acoustic_indices.csv')
    result_dict = {col: df_indices[col].values for col in df_indices.columns}
    indices = result_dict
else:
    indices = get_acoustic_indices(DATA_DIR, loader)
    indices.to_csv(Path(DATA_DIR) / 'acoustic_indices.csv')
        
df_features = get_umap_2d(DATA_DIR, bacpipe_features)

concatenate_features(DATA_DIR, df_features, indices)
