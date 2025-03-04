import h5py
import numpy as np
from PIL import Image
from typing import List
import torch
from torch.utils.data import Dataset

class DownstreamDataset(Dataset):
    def __init__(
            self,
            file_paths: str,
            columns: List[str],
            image_processor,
            ):
        super().__init__()
        self.columns = columns
        self.image_processor = image_processor
        if isinstance(file_paths, str):
            self.h5_file = h5py.File(file_paths, 'r')
        elif isinstance(file_paths, list):
            self.h5_file = self.concat_h5_files(file_paths)

    def concat_h5_files(self, file_paths):
        """
        先计算所有数据集的总行数，预先分配大数组，然后直接写入数据。
        要求各文件中同一列的数组除第一维外形状一致。
        """
        # 计算每个数据集拼接后的总行数，并记录除行之外的 shape
        total_rows = {col: 0 for col in self.columns}
        sample_shapes = {}
        for fp in file_paths:
            with h5py.File(fp, 'r') as f:
                for col in self.columns:
                    ds = f[col]
                    total_rows[col] += ds.shape[0]
                    if col not in sample_shapes:
                        sample_shapes[col] = ds.shape[1:]

        # 提取各数据集的数据类型，假设所有文件相同
        with h5py.File(file_paths[0], 'r') as f:
            dtypes = {col: f[col].dtype for col in self.columns}

        # 预分配最终数组
        data_dict = {}
        for col in self.columns:
            data_dict[col] = np.empty((total_rows[col],) + sample_shapes[col], dtype=dtypes[col])

        # 按顺序将各文件数据复制到预分配的数组中
        current_index = {col: 0 for col in self.columns}
        for fp in file_paths:
            with h5py.File(fp, 'r') as f:
                for col in self.columns:
                    data = f[col][:]
                    nrows = data.shape[0]
                    data_dict[col][current_index[col]:current_index[col] + nrows] = data
                    current_index[col] += nrows

        return data_dict


    def __len__(self):
        return len(self.h5_file[self.columns[0]])
        
    def __getitem__(self, idx):

        return_dic = {}
        for col in self.columns:
            if col == 'images':
                image_data = self.h5_file[col][idx]
                image_data = np.transpose(image_data, (1, 2, 0))
                image_data = Image.fromarray(image_data)
                return_dic[col] = self.image_processor(image_data)

            elif col == 'tokenized_gene':
                return_dic[col] = torch.from_numpy(self.h5_file[col][idx]).clone()

            elif col == 'spot_label':
                data_item = self.h5_file[col][idx]
                return_dic[col] = np.array(data_item, dtype=int)

            elif col == 'spot_names':
                return_dic[col] = self.h5_file[col][idx].decode('utf-8')


        return return_dic