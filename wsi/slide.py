# libraries
from typing import Optional, Dict, Union, List, Tuple
import openslide
import os

class WholeSlideImage:
    """ Class representing a whole-slide image (WSI) and its associated properties. """

    def __init__(self, path: Optional[str]=None):
        self.slide = None
        self.name = None
        self.contours = {}

        if path is not None:
            self.load_slide(path)

    def load_slide(self, path):
        # validate file existence
        if not os.path.exists(path):
            raise FileNotFoundError(f"No such file or directory: {path}")

        try:
            # read whole-slide-image with openslide
            self.slide = openslide.OpenSlide(path)
            self.name = os.path.splitext(os.path.basename(path))[0]
        except openslide.OpenSlideError as error:
            self.slide = None
            raise openslide.OpenSlideError(f"Failed to open slide {path}: {str(error)}")
        
    @property
    def name(self):
        return self._name
    
    @name.setter
    def name(self, name):
        self._name = name
    
    @property
    def slide(self):
        return self._slide

    @slide.setter
    def slide(self, file):
        if file is None:
            self._slide = None
        elif isinstance(file, openslide.OpenSlide):
            self._slide = file

            # cache slide properties for performance
            level_dimensions = self.slide.level_dimensions
            reference = level_dimensions[0]
            level_downsamples = []
            for dimension in level_dimensions:
                downsample = (reference[0]/dimension[0], reference[1]/dimension[1])
                level_downsamples.append(downsample)
            
            self.level_downsamples = tuple(level_downsamples)
            self.level_dimensions = level_dimensions
            self.num_levels = self.slide.level_count
        else:
            raise ValueError(f"File is not a supported slide format: {type(file)}")
    
    @property
    def project(self):
        return self._project
    
    @project.setter
    def project(self, value: str):
        if not isinstance(value, str):
            raise ValueError(f"invalid value for 'project': expected a string, got `{type(value).__name__}`.")
        self._project = value

    @property
    def contours(self):
        return self._contours

    @contours.setter
    def contours(self, contours: Dict):
        if not isinstance(contours, Dict):
            raise ValueError(f"invalid value for 'contours': expected a dict, got `{type(value).__name__}`.")
        self._contours = {
            "tissue_contours": contours.get("tissue_contours", []),
            "tissue_holes": contours.get("tissue_holes", [])
        }
    
    @property
    def num_levels(self):
        if hasattr(self, "_num_levels"):
            return self._num_levels

    @num_levels.setter
    def num_levels(self, value: int):
        if not isinstance(value, int):
            raise ValueError(f"invalid value for 'num_levels': expected an int, got `{type(value).__name__}`.")
        self._num_levels = value
        
    @property
    def level_dimensions(self):
        if hasattr(self, "_level_dimensions"):
            return self._level_dimensions

    @level_dimensions.setter
    def level_dimensions(self, value: Union[List[Tuple], Tuple[Tuple]]):
        if not isinstance(value, (Tuple, List)):
            raise ValueError(f"invalid value for 'level_dimensions': expected a list or tuple of tuples, got `{type(value).__name__}`.")
        self._level_dimensions = value

    @property
    def level_downsamples(self):
        if hasattr(self, "_level_downsamples"):
            return self._level_downsamples

    @level_downsamples.setter
    def level_downsamples(self, value: Union[List[Tuple], Tuple[Tuple]]):
        if not isinstance(value, (Tuple, List)):
            raise ValueError(f"invalid value for 'level_downsamples': expected a list or tuple of tuples, got `{type(value).__name__}`.")
        self._level_downsamples = value