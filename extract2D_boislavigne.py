import pandas as pd
import numpy as np
import json
import yaml

path = '/home/lorenzo-dubois/Documents/Thèse/tools/bacpipe/bacpipe_results/data_sample/dim_reduced_embeddings/2025-11-28_18-28___umap-data_sample-birdnet'

with open(path+'/data_sample_umap.json', 'r') as f:
    embeds = json.load(f)
    
df = pd.DataFrame()
x = embeds['x']
y = embeds['y']

file_names = []
[file_names.extend([f] * ll) for f, ll in zip(embeds['metadata']['audio_files'], embeds['metadata']['nr_embeds_per_file'])]


starts = []
# was previously 1000 
[starts.extend((np.arange(ll)*3000).tolist()) for ll in embeds['metadata']['nr_embeds_per_file']]

duration = [int((embeds['metadata']['segment_length (samples)'] / embeds['metadata']['sample_rate (Hz)']) * 1000)] * len(x)


limit = 731*20
df['starts'] = starts[:limit]
df['duration'] = duration[:limit]
df['x'] = x[:limit]
df['y'] = y[:limit]
df['file_names'] = file_names[:limit]

for file in df.file_names.unique():
    df_temp = df[df.file_names == file]
    df_temp.to_csv(
        f"/media/lorenzo-dubois/TRIM_4/files/{file.split('/')[-1].replace('wav', 'txt')}", 
        sep=' ', 
        index=False,
        header=None
        )

# df.to_csv('embeddings.csv', sep=' ', index=False)

