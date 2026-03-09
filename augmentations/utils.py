import numpy as np
import torch

def fft(image):
    # apply n-dimensional FFT across the channels
    return np.fft.fftn(image, axes=(-3, -2))

def fftshift(image):
    # shift low frequencies to center
    return np.fft.fftshift(image, axes=(-3, -2))

def ifft(image):
    return np.fft.ifftn(image, axes=(-3, -2))

def ifftshift(image):
    return np.fft.ifftshift(image, axes=(-3, -2))

def is_image(image):
    if isinstance(image, np.ndarray) and image.ndim == 3:
        return True
    else:
        return False

def is_uint8_image(image):
    if is_image(image) and image.dtype == np.uint8:
        return True
    else:
        return False

def rgb_to_od(image, Io: int=255):
    """ 
    Convert from RGB to optical density space.
    OD = -log(image / 255) 
    """
    assert is_uint8_image(image)
    
    image = image.astype(np.float32) + 1e-6
    od_image = -1 * np.log(image / Io)
    return od_image

def to_tensor(batch):
    default_float_dtype = torch.get_default_dtype()
    # handle numpy array
    if isinstance(batch, np.ndarray):
        if batch.ndim == 3:
            batch = batch[:, :, :, None]

        # backward compatibility
        batch = torch.from_numpy(np.transpose(batch, axes=(0, 3, 1, 2))).contiguous()
        
        if isinstance(batch, torch.ByteTensor):
            return batch.to(dtype=default_float_dtype).div(255)
        else:
            return batch.to(dtype=default_float_dtype)
        
def normalize(image):
    """ Normalize image values to 0-255 range. """
    if image.max() == image.min():
        return np.zeros_like(image, dtype=np.uint8)
    img_max = np.max(image, axis=(-3, -2), keepdims=True)
    img_min = np.min(image, axis=(-3, -2), keepdims=True)
    image = (image - img_min) / (img_max - img_min)
    image = (image * 255).astype(np.uint8)
    return image

def get_radial_mask(shape: tuple, radius: int=14, filter_type: str="low"):
    """
    Create a radial mask for frequency filtering.
    
    Args:
        shape (tuple): Height and width of the mask
        radius (int): Radius of the filter
        filter_type (str): Either "low" for low-pass or "high" for high-pass filter
        
    Returns:
        np.ndarray: Radial mask with specified dimensions
    """
    height, width = shape
    x_center, y_center = int(width / 2), int(height / 2)

    # create coordinate grid
    x_cords, y_cords = np.ogrid[:height, :width]

    # calculate distance from center for each point
    distance = np.sqrt((x_cords - x_center)**2 + (y_cords - y_center)**2)

    # create mask based on filter type
    if filter_type == "low":
        mask = distance <= radius
    elif filter_type == "high":
        mask = distance > radius
    else:
        raise ValueError(f"Invalid filter_type: {filter_type}. Use 'low' or 'high'.")

    return mask.astype(np.float32)



def fourier2d(image: np.ndarray, mask: np.ndarray):
    """ Apply Fourier transform to a single image or batch of images. """
    # perform fast fourier transform (FFT)
    ft_image = fft(image)
    shift_ft = fftshift(ft_image)

    # create mask and apply it
    masked_ft = np.multiply(shift_ft, mask)

    # perform inverse fourier transform (IFFT)
    ift_shift = ifftshift(masked_ft)
    ift_patch = ifft(ift_shift)

    # normalize the transformed image
    fourier_image = normalize(np.real(ift_patch))
    
    return fourier_image