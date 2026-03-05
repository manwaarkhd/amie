from torchvision import transforms
from easydict import EasyDict

config = EasyDict()

config.CANCER_TYPES = [
    "BLCA",  # Bladder Urothelial Carcinoma
    "BRCA",  # Breast Invasive Carcinoma
    "ESCA",  # Esophageal Carcinoma
    "GBM",   # Glioblastoma Multiforme
    "HNSC",  # Head and Neck Squamous Cell Carcinoma
    "LGG",   # Brain Lower Grade Glioma
    "LUAD",  # Lung Adenocarcinoma
    "LUSC",  # Lung Squamous Cell Carcinoma
    "SARC",  # Sarcoma
    "STAD",  # Stomach Adenocarcinoma
    "SKCM",  # Skin Cutaneous Melanoma
    "UCEC",  # Uterine Corpus Endometrial Carcinoma
]

# downsample factors for WSI pyramid levels
config.LEVEL_DOWNSAMPLES = [1, 4, 16, 32]

# permitted WSI file extensions
config.WSI_EXTENSIONS = {".svs", ".ndpi", ".tiff", ".tif", ".mrxs", ".scn"}

cfg = config  # alias