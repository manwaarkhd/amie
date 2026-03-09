from sklearn.model_selection import StratifiedGroupKFold
from typing import List, Optional, Union
import pandas as pd
import os

from wsi import WholeSlideImage
from config import cfg

class Dataset:
    """Base dataset class for Whole Slide Image datasets."""

    def __init__(self, path: str, projects: Union[str, List[str]]):
        if not isinstance(projects, (str, list)):
            raise TypeError(
                f"invalid dtype for `projects`. Expected 'str' or 'list', but got {type(projects).__name__}."
            )

        # normalize project(s) to a list
        if isinstance(projects, str):
            if projects.lower() == "all":
                projects = cfg.CANCER_TYPES
            else:
                projects = [projects]

        # validate project names
        invalid = set(projects) - set(cfg.CANCER_TYPES)
        if invalid:
            raise ValueError(f"invalid project(s): {invalid}. Allowed: {cfg.CANCER_TYPES}")

        self.projects = projects
        self.path = path

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

class TCGA(Dataset):
    """ TCGA dataset class for Whole Slide Image classification."""
    
    def __init__(
        self,
        path: str,
        annotations: Union[str, pd.DataFrame],
        projects: Union[str, List[str]]="all",
        split: Optional[str]="train"
    ):
        super(TCGA, self).__init__(path, projects)
        if not os.path.isdir(self.path):
            raise FileNotFoundError(f"path does not exist or is not a directory: '{self.path}'")
            
        if isinstance(annotations, str):  # path to CSV
            if not os.path.isfile(annotations):
                raise FileNotFoundError(f"annotation file not found: {annotations}")
            annotations = pd.read_csv(annotations)
        elif isinstance(annotations, pd.DataFrame):  # already loaded DataFrame
            annotations = annotations.copy()
        else:
            raise ValueError("`annotations` must be a file path or a DataFrame.")
        
        # filter by projects
        annotations = annotations[annotations["project"].isin(self.projects)]
        if annotations.empty:
            raise ValueError("no samples found for the specified projects = {self.projects}")
        
        # filter by split
        if split is not None:
            if split not in ["train", "test"]:
                raise ValueError(f"invalid split: '{split}'. Expected 'train', 'test', or None.")
            if "split" not in annotations.columns:
                raise ValueError("`annotations` must contain a 'split' column when `split` is specified.")
            annotations = annotations[annotations["split"] == split].reset_index(drop=True)

        # prepare samples list
        self.samples = [
            {
                "slide_name": os.path.splitext(record["slide_name"])[0],
                "project": record["project"],
                "label": record["label"],
                "format": os.path.splitext(record["slide_name"])[1],
            }
            for _, record in annotations.iterrows()
        ]

    def __getitem__(self, index: int):
        """ Return the (WholeSlideImage, label) for the given index. """
        sample = self.samples[index]
        slide_name = sample["slide_name"] + sample["format"]
        project = sample["project"]
        
        path = os.path.join(self.path, project, "slides", slide_name)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"slide not found: {path}")
        
        wsi = WholeSlideImage(path)
        wsi.project = project
        label = sample["label"]
            
        return wsi, label

    def __len__(self):
        """ Return the number of samples. """
        return len(self.samples)
    
class TCGAKFold(Dataset):
    """ TCGA dataset wrapper for K-Fold cross-validation. """
    
    def __init__(
            self, 
            path: str, 
            annotations: Union[str, pd.DataFrame], 
            projects: Union[str, List[str]]="all", 
            num_folds: int=3, 
            random_state: int=42
        ):
        super(TCGAKFold, self).__init__(path, projects)
        if not os.path.isdir(self.path):
            raise FileNotFoundError(f"path does not exist or is not a directory: '{self.path}'")
        
        if isinstance(annotations, str):  # path to CSV
            if not os.path.isfile(annotations):
                raise FileNotFoundError(f"annotation file not found: {annotations}")
            annotations = pd.read_csv(annotations)
        elif isinstance(annotations, pd.DataFrame):  # already loaded DataFrame
            annotations = annotations.copy()
        else:
            raise ValueError("`annotations` must be a file path or a DataFrame.")

        # filter by projects
        annotations = annotations[annotations["project"].isin(self.projects)]
        if annotations.empty:
            raise ValueError("no samples found for the specified projects = {self.projects}")

        # initialize the splitter
        sgkf = StratifiedGroupKFold(num_folds, shuffle=True, random_state=random_state)

        # create folds
        folds_data = []
        for project in self.projects:
            dataframe = annotations[annotations["project"] == project]
            dataframe = dataframe.reset_index(drop=True)
            
            groups = dataframe["patient_id"]
            X = dataframe["slide_name"]
            y = dataframe["label"]
            
            for fold, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups)):
                train_df = dataframe.iloc[train_idx].copy()
                train_df["split"] = "train"
                train_df["fold"] = fold
                folds_data.append(train_df)
        
                test_df = dataframe.iloc[test_idx].copy()
                test_df["split"] = "test"
                test_df["fold"] = fold
                folds_data.append(test_df)
        folds_data = pd.concat(folds_data, ignore_index=True)

        # prepare folds
        self.folds = []
        for fold in range(num_folds):
            annotations = folds_data[folds_data["fold"] == fold].drop(columns=["fold"])            
            self.folds.append({
                "train": TCGA(path, annotations, projects, split="train"),
                "test": TCGA(path, annotations, projects, split="test"),
            })
        
    def __getitem__(self, index):
        """ Return the (train, test) TCGA datasets for the given fold index. """
        return self.folds[index]

    def __len__(self):
        """ Return the number of folds. """
        return len(self.folds)

    def summary(self):
        """ Print a summary of sample distribution across folds. """
        def count_labels(dataset, label):
            count = sum(1 for sample in dataset.samples if sample['label'] == label)
            return count
        
        for index, fold in enumerate(self.folds):
            train_data, test_data = fold["train"], fold["test"]
            
            print(f"Fold {index + 1}:")
            print(f"  Train Slides: {len(train_data)}")
            print(f"    Positives: {count_labels(train_data, label=1)}")
            print(f"    Negatives: {count_labels(train_data, label=0)}\n")

            print(f"  Test Slides: {len(test_data)}")
            print(f"    Positives: {count_labels(test_data, label=1)}")
            print(f"    Negatives: {count_labels(test_data, label=0)}\n")