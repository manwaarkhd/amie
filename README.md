<h2 align="center"> Detecting Extrachromosomal DNA from Routine Histopathology </h2>
<p align="justify">
Extrachromosomal DNA (ecDNA) is a major driver of oncogene amplification, tumour heterogeneity and poor clinical outcomes, yet its detection relies on specialised genomic assays that are not integrated into routine diagnostics. Here, we show that ecDNA status can be inferred directly from standard haematoxylin and eosin–stained whole-slide pathology images. We develop an end-to-end, weakly supervised deep learning framework that aggregates thousands of high-magnification patches per slide with slide-level augmentation and interpretable attention. Across twelve cancer types from The Cancer Genome Atlas, the approach identifies tumours with genomic amplifications and, critically, distinguishes ecDNA-amplified from chromosomally amplified or non-amplified tumours, with the strongest signal in glioblastoma. Attention maps localise regions enriched for nuclei with altered chromatin intensity and texture, and predicted ecDNA status recapitulates its adverse association with survival. These results indicate that ecDNA amplifications leave reproducible histomorphologic foot-prints detectable by routine pathology, enabling scalable screening to prioritise tumours for confirmatory molecular testing.
</p>

## Updates
- **02/03/2026**: Live on biorxiv.

# Usage Guide
### Prerequisites
- Python 3.8
- TensorFlow 2.10
- OpenCV 4.9
### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/manwaarkhd/amie.git
   cd amie
   ```
3. Set up a Python virtual environment (optional but recommended):
   ```bash
   python3 -m venv env
   source env/bin/activate  # On Windows use `env\Scripts\activate`
   ```
5. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

# Acknowledgements
The authors acknowledge the support of the Ministry of Science and Culture of Lower Saxony through funds from the program zukunft.niedersachsen of the Volkswagen Foundation for the `CAIMed – Lower Saxony Center for Artificial Intelligence and Causal Methods in Medicine' project (grant no. ZN4257). The authors acknowledge Hannover Medical School for providing MHH-HPC resources and technical support that have contributed to the research results reported within this paper. 

# Citation
If you find our work useful in your research, please consider citing our paper:
```BibTeX
@article {Khalid2026.02.27.708546,
	author = {Khalid, Muhammad Anwaar and Gratius, Michael and Brown, Christopher and Younis, Raneen and Ahmadi, Zahra and Chavez, Lukas},
	title = {Detecting Extrachromosomal DNA from Routine Histopathology},
	year = {2026},
	doi = {10.64898/2026.02.27.708546},
	publisher = {Cold Spring Harbor Laboratory},
	journal = {bioRxiv}
}
```


