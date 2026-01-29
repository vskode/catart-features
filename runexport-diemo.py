
from generate_catart_files import get_bacpipe_features, get_umap_2d, concatenate_features
from audio_prep import get_acoustic_indices

# global config
MODELS = ['birdnet', 'perch_bird', 'beats', 'naturebeats']
DATA_DIR = '/Users/audio/sounds/nature/Bois_Lavigne/Bois_Lavigne_04_2024'

bacpipe_features, audio_path = get_bacpipe_features(MODELS, DATA_DIR)

indices = get_acoustic_indices(audio_path)
        
df_features = get_umap_2d(bacpipe_features)

concatenate_features(df_features, indices)
