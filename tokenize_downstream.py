from models._utils import sf_normalize, tokenize_data, set_seed
from scipy.sparse import csr_matrix
import scipy.sparse as sp
from PIL import Image
import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import h5py
import pickle
import os


label_mapping = {
    'DLPFC':{
        '151676': 'Human_Brain_Maynard_02082021_Visium_151676',
        '151675': 'Human_Brain_Maynard_02082021_Visium_151675',
        '151674': 'Human_Brain_Maynard_02082021_Visium_151674',
        '151673': 'Human_Brain_Maynard_02082021_Visium_151673',
        '151672': 'Human_Brain_Maynard_02082021_Visium_151672',
        '151671': 'Human_Brain_Maynard_02082021_Visium_151671',
        '151670': 'Human_Brain_Maynard_02082021_Visium_151670',
        '151669': 'Human_Brain_Maynard_02082021_Visium_151669',
        '151510': 'Human_Brain_Maynard_02082021_Visium_151510',
        '151509': 'Human_Brain_Maynard_02082021_Visium_151509',
        '151508': 'Human_Brain_Maynard_02082021_Visium_151508',
        '151507': 'Human_Brain_Maynard_02082021_Visium_151507'
    },
}


ROOT_DIR = 'path/to/h5ad/and/patches'
OUT_PATH = './downstream_tokens'
LABEL_PATH = 'path/to/label'


save_dir = f'{OUT_PATH}/'
os.makedirs(save_dir, exist_ok=True)

set_seed(42)
# 读取 gene name id 字典和 h5ad 头
with open('gene_name_id_dict.pkl', 'rb') as f:
        gene_name_id_dict = pickle.load(f)
h5ad_head = sc.read_h5ad('./model.h5ad')
h5ad_head_vars = h5ad_head.var_names

# 读取均值
print('Reading mean expression value for each gene...')
mean = np.load('Visium_mean.npy')
mean = np.nan_to_num(mean)
rounded_values = np.where((mean % 1) >= 0.5, np.ceil(mean), np.floor(mean))
mean = np.where(mean == 0, 1, rounded_values)
print('Done!')
print('\n')

projects = list(label_mapping.keys())
for project in projects:
    label_dict = {}
    print(f"Project: {project}")
    for sample_id, label_id in label_mapping[project].items():
        print(f"Sample ID: {sample_id}, Label ID: {label_id}")
        labels = pd.read_csv(f'{LABEL_PATH}/{label_id}_anno.csv')
        if len(labels.columns) == 2:
            labels.columns = ['spot_id', 'label']
        else:
            labels.columns = ['spot_id', 'label', '']
        labels = labels.set_index('spot_id')
        adata = sc.read_h5ad(f'{ROOT_DIR}/{sample_id}.h5ad')
        # 稀疏矩阵转换
        if not sp.isspmatrix_csr(adata.X):
            if sp.isspmatrix(adata.X):  # 检查是否为其他稀疏矩阵类型
                adata.X = adata.X.tocsr()  # 转换为CSR格式
            elif isinstance(adata.X, np.ndarray):  # 检查是否为numpy数组
                adata.X = csr_matrix(adata.X)  # 将numpy数组转换为CSR格式
        print(f'The ori shape is {adata.X.shape}')
        print('\n')

        # 添加label
        adata.obs['label'] = None
        for spot_id in adata.obs.index:
            lable_spot_id = f'{label_id}_{spot_id}'
            if lable_spot_id in labels.index:
                spot_label = labels.loc[lable_spot_id]['label']
                if spot_label not in label_dict:
                    label_dict[spot_label] = len(label_dict)
                adata.obs.loc[spot_id, 'label'] = label_dict[spot_label]
        adata = adata[adata.obs['label'].notnull()].copy()
        print(f'The shape after adding label is {adata.X.shape}')
        print('\n')

        # 替换gene index
        current_var_names = adata.var_names
        new_var_names = list(current_var_names.copy())
        num_transition = 0
        for i, gene_name in enumerate(current_var_names):
            if gene_name in gene_name_id_dict:
                new_var_names[i] = gene_name_id_dict[gene_name]  # 替换为 gene_id
                num_transition += 1
        adata.var_names = new_var_names
        if adata.var_names.duplicated().any():
            adata.var_names_make_unique()
        trans_percent = num_transition/len(new_var_names) * 100

        # 为obs索引添加信息，防止出现重复
        current_obs_names = adata.obs.index
        new_obs_names = [f"{sample_id}_{obs}" for obs in current_obs_names]
        adata.obs.index = new_obs_names

        # 清除词表中不存在的var
        adata_filtered = adata[:, adata.var_names.isin(h5ad_head_vars)].copy()
        assert adata_filtered.var_names.is_unique
        adata = ad.concat([h5ad_head, adata_filtered], join='outer', axis=0)
        adata = adata[1:].copy()
        print(f'The processed shape is {adata.shape}')
        print(f'Transform {trans_percent:.2f}% gene name to gene id.')
        print('\n')
        
        # 清楚没有对应 image 的 spot
        print('Filter spots by images...')
        image_files = {file_name[:-4] for file_name in os.listdir(f'{ROOT_DIR}/patches_image') if file_name.endswith('.png')}
        spot_ids = set(adata.obs.index)
        missing_spots = spot_ids - image_files
        adata = adata[~adata.obs.index.isin(missing_spots)]
        print(f'The adata after image filter has {adata.n_obs} spots')
        print('Done!\n')

        adata.obs['spot_id'] = adata.obs.index
        adata.obs.reset_index(drop=True, inplace=True)
        adata_obs = adata.obs
        
        adata_obs = adata_obs.reset_index().rename(columns={'index':'idx'})
        adata_obs['idx'] = adata_obs['idx'].astype('i8')

        # Tokenize
        x = adata.X
        x = np.nan_to_num(x) # is NaN values, fill with 0s
        x = sf_normalize(x)
        median_counts_per_gene = mean
        median_counts_per_gene += median_counts_per_gene == 0
        out = x / median_counts_per_gene.reshape((1, -1))
        tokenized_idx = tokenize_data(out, 4096, 30)

        # 加载数据
        image_arrays = []
        spot_names = []
        for i in range(len(adata_obs)):
            spot_id = adata_obs['spot_id'][i]
            img = Image.open(f'{ROOT_DIR}/patches_image/{spot_id}.png')
            img = img.convert('RGB')
            img_array = np.array(img)# (224,224,3)
            image_arrays.append(img_array) 
            spot_names.append(spot_id)
        image_arrays = np.stack(image_arrays)
        image_arrays = np.transpose(image_arrays, (0, 3, 1, 2))

        with h5py.File(f'{save_dir}/{sample_id}.h5', 'w') as f:
            f.create_dataset('tokenized_gene', data=tokenized_idx)
            f.create_dataset('images', data=image_arrays)
            f.create_dataset('spot_label', data=np.array(adata.obs['label'], dtype=int))
            f.create_dataset('spot_names', data=np.array(spot_names, dtype='S'))