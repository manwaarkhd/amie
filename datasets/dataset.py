from torch.utils.data import Dataset
import numpy as np
import cv2
import os

from .utils import to_tensor, haskey, get_attributes, load_coordinates
from wsi.utils import is_valid_patch

class WSIBags(Dataset):
    """ A baseline PyTorch Dataset for WSI bags. """
    
    def __init__(self, dataset: Dataset, level: int):
        self.dataset = dataset
        self.samples = self.dataset.samples
        self.level = level
    
    def _extract_patches(self, wsi, coordinates, **kwargs):
        """ Extract patches from WSI at specified coordinates. """
        # check if the target downsample level exists in the WSI
        downsamples = [int(downsample[0]) for downsample in wsi.level_downsamples]
        target_downsample = kwargs["downsample"]
        use_existing_level = (
            target_downsample in downsamples
            and self.level == downsamples.index(target_downsample)
        )
        
        # determine patch extraction parameters
        level = self.level if use_existing_level else 0
        patch_size = kwargs["patch_size"] if use_existing_level else kwargs["patch_size"] * target_downsample
        resize = not use_existing_level

        # extract patches
        patches = []
        for (x, y) in coordinates:
            try:
                patch = wsi.slide.read_region((x, y), level=level, size=(patch_size, patch_size)).convert("RGB")
                patch = np.array(patch)

                if resize:
                    patch = cv2.resize(
                        np.array(patch), 
                        None, 
                        fx=(1.0 / target_downsample), 
                        fy=(1.0 / target_downsample), 
                        interpolation=cv2.INTER_LANCZOS4
                    )
                
                if is_valid_patch(patch, threshold=0.1):
                    patches.append(patch)
            except Exception as exp:
                print(f"WARNING: Failed to read patch at ({x}, {y}): {exp}")
                continue
        
        if not patches:
            raise RuntimeError(f"no valid patches extracted for slide {wsi.name}")
        patches = np.stack(patches, axis=0)
        
        return patches 

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
    

class TCGABags(WSIBags):
    """ A PyTorch Dataset for TCGA WSI bags. """

    def __init__(
        self,
        dataset: Dataset,
        level: int,
        augmentations: list=None
    ):
        super(TCGABags, self).__init__(dataset, level)

        identity = lambda bag: bag
        if augmentations is None:
            augmentations = [identity]
        else:
            augmentations = [identity] + augmentations
        
        self.augmentations = np.array(augmentations, dtype=object)
        self.augmentation_map = np.repeat(
            self.augmentations,
            repeats=np.repeat(len(self.dataset), len(self.augmentations)),
            axis=0
        )
        
    def _load_patches(self, path: str, group_key: str):
        """ Load patches or coordinates from an HDF5 file. """
        if haskey(path, group_key, search_key="patches"):
            generator = load_coordinates(path, group_key, coordinates_only=False, batch_size=1)
            coordinates, patches = zip(*generator)
            coordinates, patches = np.vstack(coordinates), np.vstack(patches)
            return coordinates, patches
        elif haskey(path, group_key, search_key="coordinates"):
            generator = load_coordinates(path, group_key, coordinates_only=True, batch_size=1)
            coordinates = np.vstack([coordinate for coordinate in generator])
            return coordinates, None
        else:
            raise KeyError(f"neither 'patches' nor 'coordinates' dataset found in group '{group_key}' of file '{os.path.basename(path)}'")
        
    def __getitem__(self, index):
        """ Return the (patches, label) for the given index. """
        sample_idx = index % len(self.dataset)
        wsi, label = self.dataset[sample_idx]
        group_key = str(self.level)

        folder = os.path.join(self.dataset.path, wsi.project, "patches")
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"patch folder not found: {folder}")
        
        file = os.path.join(folder, wsi.name + ".h5")
        if not os.path.isfile(file):
            raise FileNotFoundError(f"file '{wsi.name}.h5' not found in {folder}")

        # load patches or coordinates from HDF5 file
        coordinates, patches = self._load_patches(file, group_key)

        # if only coordinates are available, extract patches from WSI
        if patches is None:
            attrs = get_attributes(file, group_key)
            patches = self._extract_patches(wsi, coordinates, attrs)

        augmentation = self.augmentation_map[index]
        bag = augmentation(patches)
        bag = to_tensor(bag)

        return bag, label

    def __len__(self):
        """ Return the number of samples in the dataset. """
        return len(self.dataset) * len(self.augmentations) 