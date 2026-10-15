
from generate_catart_files import get_bacpipe_features, get_umap_2d, concatenate_features
from audio_prep import get_acoustic_indices

# global config
MODELS = ['perch_v2']
# DATA_DIR = '/Users/audio/sounds/nature/Bois_Lavigne/Bois_Lavigne_04_2024'
DATA_DIR = '/home/siriussound/Code/bacpipe/bacpipe/tests/test_data/audio'

bacpipe_features, audio_path, loader = get_bacpipe_features(DATA_DIR, MODELS)

indices = get_acoustic_indices(audio_path, loader)
        
df_features = get_umap_2d(DATA_DIR, bacpipe_features)

concatenate_features(DATA_DIR, df_features, indices)
