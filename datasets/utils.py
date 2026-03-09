import numpy as np
import pickle
import torch
import h5py

def to_tensor(batch):
    """ Convert a numpy array (patches or features) to a torch.Tensor. """
    default_float_dtype = torch.get_default_dtype()

    if isinstance(batch, np.ndarray):
        if batch.ndim == 3:  
            # single patch [H, W, C] -> add batch dim
            batch = batch[None, ...]
        
        if batch.ndim == 4:  
            # patches [N, H, W, C] -> [N, C, H, W]
            batch = torch.from_numpy(np.transpose(batch, (0, 3, 1, 2))).contiguous()
        elif batch.ndim == 2:  
            # features [N, D], no transpose needed
            batch = torch.from_numpy(batch).contiguous()
        else:
            raise ValueError(f"Unsupported numpy shape {batch.shape} for to_tensor")

        # normalize if uint8 image patches
        if isinstance(batch, torch.ByteTensor):
            return batch.to(dtype=default_float_dtype).div(255)
        else:
            return batch.to(dtype=default_float_dtype)

    elif isinstance(batch, torch.Tensor):
        # already a tensor: ensure float dtype
        return batch.to(dtype=default_float_dtype)

    else:
        raise TypeError(f"Unsupported input type {type(batch)} for to_tensor")

def is_empty(path: str, group_key: str, dataset_key: str):
    """ Check whether a dataset inside a given group in an HDF5 file is empty. """
    try:
        with h5py.File(path, "r") as file:
            if group_key not in file:
                raise KeyError(f"Group '{group_key}' not found in file '{path}'.")
            group = file[group_key]

            if dataset_key not in group:
                raise KeyError(f"Dataset '{dataset_key}' not found in group '{group_key}'.")
            
            dataset = group[dataset_key]
            return dataset.shape[0] == 0  # empty if first dimension is 0
    except OSError as e:
        raise OSError(f"Unable to open HDF5 file '{path}': {e}")
    
def get_attributes(path: str, group_key: str):
    """ Retrieve all attributes of a given group in an HDF5 file. """
    try:
        with h5py.File(path, "r") as file:
            if group_key not in file:
                raise KeyError(f"Group '{group_key}' not found in file '{path}'.")
            
            group = file[group_key]
            return {key: group.attrs[key] for key in group.attrs}
    except OSError as e:
        raise OSError(f"Unable to open HDF5 file '{path}': {e}")

def haskey(path: str, group_key: str, search_key: str):
    """ Check if a search_key exists inside the specified group of an HDF5 file. """
    try:
        with h5py.File(path, "r") as file:
            if group_key not in file:
                raise KeyError(f"Group '{group_key}' not found in file.")
            group = file[group_key]
            return search_key in group
    except OSError as e:
        raise OSError(f"Unable to open HDF5 file '{path}': {e}")

def load_pickle(path):
    data = None
    try:
        file = open(path, mode="rb")
        data = pickle.load(file)
        file.close()
    except Exception as error:
        print(error)
    return data

def load_coordinates(path: str, group_key: str, mode: str="valid", coordinates_only: bool=True, batch_size: int=None):
    file = h5py.File(path, "r")
    if group_key not in file:
        file.close()
        raise KeyError(f"group '{group_key}' not found in file.")
    
    group = file.get(group_key)    
    coordinates = group["coordinates"]
    
    if mode == "valid":
        indices = np.where(coordinates[:, -1] == 1)[0]  
    elif mode == "all":
        indices = np.arange(len(coordinates))
    else:
        raise ValueError
    
    coordinates = coordinates[:, :-1]
    if coordinates_only:
        file.close()
        return coordinates[indices]

    def generator(coordinates, patches, indices):
        num_samples = len(indices)
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch_idx = indices[start:end]
            yield coordinates[batch_idx], patches[batch_idx]
        file.close()

    if not is_empty(path, group_key, dataset_key="patches"):
        patches = group["patches"]
        if batch_size is None:
            output = coordinates[indices], patches[indices]
            file.close()
            return output
        else:
            return generator(coordinates, patches, indices)
    else:
        file.close()
        raise KeyError(KeyError(f"Dataset 'patches' not found in group '{group_key}'."))