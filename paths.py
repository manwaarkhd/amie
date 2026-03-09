import os

# get the root directory
root = os.path.dirname(__file__)

class Paths(object):
    """ A class to manage directory paths. """
        
    @classmethod
    def get_dataset_path(cls, name: str):      
        if name.lower() == "tcga":
            return os.path.join(root, "datasets", "TCGA")
        else:
            raise ValueError(f"`{name}` dataset doesn't exist in your paths.py file")
    
    @classmethod
    def get_ckpt_path(cls):
        return os.path.join(root, "logs")