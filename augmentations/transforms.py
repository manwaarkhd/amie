import torchvision.transforms as transforms
from typing import Optional, Union, Tuple
from PIL import Image
import numpy as np

from .utils import get_radial_mask, fourier2d

class RandomPatchMask:
    """
    Randomly masks a percentage of patches in a bag by setting their pixel values to a specified mask value.
    
    Attributes:
        strength (tuple): Range (min, max) for the percentage of patches to mask (default: (0.25, 0.75)).
        value (int, float): The value to assign to masked patches (default: 0).
        mode (str or None): Type of masking to apply. If None, a random mode is selected for each call. Available modes: "constant", "mean", or "hybrid".
        grid_size (int): Grid size used in the hybrid masking strategy.
    """
    # available modes
    modes = ["constant", "mean", "hybrid"]

    def __init__(self, strength: Union[float, Tuple[float, float]]=(0.0, 0.25), value: Union[int, float]=0, mode: Optional[str]=None, **kwargs):
        if isinstance(strength, float):
            if not (0.0 <= strength <= 1.0):
                raise ValueError(f"Invalid strength value: {strength}. Must be between 0 and 1.")
            self.strength = (0.0, strength)
        elif isinstance(strength, (tuple, list)) and len(strength) == 2:
            if not (0.0 <= strength[0] <= strength[1] <= 1.0):
                raise ValueError(f"Invalid strength range: {strength}. Must be between 0 and 1 with min <= max.")
            self.strength = tuple(strength)
        else:
            raise TypeError(f"Invalid dtype for strength: {type(strength)}. Must be a float or a tuple/list of two floats.")

        if (mode is not None) and (mode not in self.modes):
            raise ValueError(f"Invalid mode: '{mode}'. Must be one of {self.modes}.")
        
        self.grid_size = kwargs.get("grid_size", 16)
        self.value = value
        self.mode = mode

    def __call__(self, bag: np.ndarray) -> np.ndarray:
        if not isinstance(bag, np.ndarray) or bag.ndim != 4:
            raise ValueError("Input must be a 4D NumPy array: (num_patches, height, width, channels).")
        # extract bag dimensions  
        num_patches, height, width, channels = bag.shape

        # randomly select the percentage of patches to mask
        percentage = np.random.uniform(low=self.strength[0], high=self.strength[1])
        
        # randomly select patches to be masked
        indices = np.random.choice(np.arange(num_patches), size=max(1, int(percentage * num_patches)), replace=False)

        # select masking mode (random if None, otherwise use specified mode)
        current_mode = self.mode or np.random.choice(self.modes)
    
        # mask selected patches
        if current_mode == "constant":
            bag[indices] = self.value
        elif current_mode == "mean":
            patch_means = np.mean(bag[indices], axis=(1, 2), keepdims=True) # compute mean for selected patches across channels
            bag[indices] = patch_means
        elif current_mode == "hybrid":
            # generate grid coordinates for sub-patches
            x = np.arange(0, width  - self.grid_size + 1, self.grid_size)
            y = np.arange(0, height - self.grid_size + 1, self.grid_size)
            coordinates = np.array([np.meshgrid(x, y, indexing="ij")]).reshape(2, -1).transpose()
            
            # total number of possible sub-patches
            num_grids = len(coordinates)
            
            for index in indices:
                percentage = np.random.uniform(low=0.15, high=0.85)
                grid_indices = np.random.choice(num_grids, size=int(percentage * num_grids), replace=False)
                for (x, y) in coordinates[grid_indices]:
                    bag[index, y:y+self.grid_size, x:x+self.grid_size] = self.value

        return bag


class RandomFourierTransform:
    """
    Applies Fourier transform with a random radius filter to a bag of patches.
    
    Attributes:
        filter_type (str): Type of frequency filter to apply. Options: "low" "high", "band", "notch"
        radius_range (tuple): Range of radius values (min, max) to sample from
        band_width (int): Width of band for band-pass/notch filters
    """
    filters = ["low", "high", "band", "notch"]

    def __init__(self, radius_range: Tuple[int, int]=(42, 112), filter_type: str=None, **kwargs):
        if (filter_type is not None) and (filter_type not in self.filters):
            raise ValueError(f"Invalid filter_type: '{filter_type}'. Must be one of {self.filters}.")
        self.filter_type = filter_type
        
        if isinstance(radius_range, (tuple, list)) and len(radius_range) == 2:
            if not all(isinstance(radius, int) for radius in radius_range):
                raise ValueError("Invalid radius_range: {radius_range}. Both values in range must be integers.")
            if (radius_range[0] <= 0) or (radius_range[1] <= 0):
                raise ValueError("Invalid radius_range: {radius_range}. Both values in range must be non-negative.")
            if radius_range[0] > radius_range[1]:
                raise ValueError("Invalid radius_range: {radius_range}. Both values in range must satisfy min < max.")
            self.radius_range = tuple(radius_range)
        else:
            raise TypeError(f"Invalid dtype for radius_range: {type(radius_range)}. Must be a tuple/list of two non-negative ints.")
        
        self.band_width = kwargs.get("band_width", 10)

    def __call__(self, bag):
        if not isinstance(bag, np.ndarray) or bag.ndim != 4:
            raise ValueError("Input must be a 4D NumPy array: (num_patches, height, width, channels).")
        # extract bag dimensions  
        num_patches, height, width, channels = bag.shape
        
        # select filter type (random if None, otherwise use specified filter type)
        current_filter = self.filter_type or np.random.choice(self.filters)

        # bag-level radius
        radius = np.random.randint(low=self.radius_range[0], high=self.radius_range[1])
        mask = get_radial_mask((height, width), radius, current_filter)
        mask = np.repeat(mask[:, :, np.newaxis], repeats=3, axis=-1)

        # apply transformation
        chunk_size = 256
        for start in range(0, num_patches, chunk_size):
            end = start + chunk_size
            batch = bag[start:end]
            bag[start:end] = fourier2d(batch, mask)
        
        return bag

class RandomColorDistortion:
    """
    Applies color distortion to a random subset of patches in a WSI bag.
    
    Args:
        strength (float or tuple): Float (0.0–1.0) or tuple (min, max) defining the % of patches to augment..
        brightness, contrast, saturation, hue (float): Hyperparameters for color jitter.
        s (float): Scaling factor for all parameters.
    """
    
    def __init__(self, strength: Union[float, Tuple[float, float]]=(0.0, 0.25), **kwargs):
        if isinstance(strength, float):
            if not (0.0 <= strength <= 1.0):
                raise ValueError(f"Invalid strength value: {strength}. Must be between 0 and 1.")
            self.strength = (0.0, strength)
        elif isinstance(strength, (tuple, list)) and len(strength) == 2:
            if not (0.0 <= strength[0] <= strength[1] <= 1.0):
                raise ValueError(f"Invalid strength range: {strength}. Must be between 0 and 1 with min <= max.")
            self.strength = tuple(strength)
        else:
            raise TypeError(f"Invalid dtype for strength: {type(strength)}. Must be a float or a tuple/list of two floats.")

        s = kwargs.get("s", 1.0)
        jitter_params = {
            "brightness": kwargs.get("brightness", 0.6) * s,
            "contrast":   kwargs.get("contrast",   0.6) * s,
            "saturation": kwargs.get("saturation", 0.8) * s,
            "hue":        kwargs.get("hue",        0.15) * s
        }
        
        self.transform = transforms.Compose([
            transforms.RandomApply([
                transforms.ColorJitter(**jitter_params)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2)
        ])

    def __call__(self, bag):
        if not isinstance(bag, np.ndarray) or bag.ndim != 4:
            raise ValueError("Input must be a 4D NumPy array: (num_patches, height, width, channels).")
        # extract bag dimensions  
        num_patches, height, width, channels = bag.shape

        # randomly select the percentage of patches to mask
        percentage = np.random.uniform(low=self.strength[0], high=self.strength[1])
        
        # randomly select patches to be masked
        indices = np.random.choice(np.arange(num_patches), size=max(1, int(percentage * num_patches)), replace=False)

        for index in indices:
            patch = Image.fromarray(bag[index])
            bag[index] = np.array(self.transform(patch))
        
        return bag