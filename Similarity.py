import pandas as pd
import numpy as np
from osgeo_utils.samples.build_jp2_from_xml import marker_map
from scipy.spatial.distance import pdist, squareform, braycurtis, jaccard, cosine

files = {
    'A': './data/abundance_Cirrhosis.csv',
    'M': './data/marker_Cirrhosis.csv'
}

metrics = {
    'braycurtis': braycurtis,
    'jaccard': jaccard,
    'cosine': cosine
}

for key, file_path in files.items():

    omics_data = pd.read_csv(file_path, header=0, index_col=None)
    print(f"Processing {key} data...")

    omics_data.rename(columns={omics_data.columns.tolist()[0]: 'Sample'}, inplace=True)
    omics_data.sort_values(by='Sample', ascending=True, inplace=True)

    for metric_name, metric in metrics.items():
        print(f'Compute {metric_name} similarity matrix for {key}...')

        distance_matrix = pdist(omics_data.iloc[:, 1:].values.astype(float), metric=metric)
        # 转换为相似性矩阵 (1 - distance)
        similarity_matrix = 1 - squareform(distance_matrix)

        similarity_df = pd.DataFrame(similarity_matrix, index=omics_data['Sample'], columns=omics_data['Sample'])
        similarity_df.to_csv(f'./Similarity/similarity_matrix_{metric_name}_Cirrhosis_{key}.csv', header=True, index=True)

print("All similarity matrices have been computed and saved.")

