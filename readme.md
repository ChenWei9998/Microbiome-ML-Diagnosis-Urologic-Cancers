
# Comprehensive Performance Evaluation of Machine Learning Models in Microbial Diagnosis of Urologic Cancers

## Authors

- **Luo Lei**  
- **Wang Zhankun**  
- **Guan Fengju**  
- **Li Bin**  
- **Ma Xiaocheng**  
- **Chen Wei** *(Corresponding Author)*  
  Email: [begin121@163.com](mailto:begin121@163.com)  

## Affiliation

1. Urological Department, The Affiliated Hospital of Qingdao University, Qingdao, 266000  
2. Operating Room, The Affiliated Hospital of Qingdao University, Qingdao, 266000  

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Installation

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)


### Create a Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Prepare Your Data

Ensure that your feature data and metadata are in CSV format and placed in the project directory. Update the file paths in the `main.py` script if necessary.

- **Feature Data**: e.g., `taxa.genus.Abd.csv`, `taxa.species.Abd.csv`
- **Metadata**: e.g., `group1_hc_rcc.csv`, `group2_hc_blca.csv`

### Run the Script

```bash
python main.py
```

This will execute the training and evaluation of the specified machine learning models on the provided datasets. The results, including ROC curves and confusion matrices, will be saved in the designated output directories.

## Project Structure

```
your-repo-name/
│
├── main.py
├── README.md
├── requirements.txt
├── taxa.genus.Abd.csv
├── taxa.species.Abd.csv
├── group1_hc_rcc.csv
├── group2_hc_blca.csv
├── genus_rcc/
│   ├── Logistic_roc_curve.png
│   ├── Logistic_roc_data.xlsx
│   ├── ...
├── species_rcc/
│   ├── ...
├── species_blca/
│   ├── ...
└── genus_blca/
    ├── ...
```

## Results

Upon running the script, the following outputs will be generated for each model and dataset:

- **ROC Curves**: Saved as `.png` images in the respective output directories.
- **ROC Data**: Saved as `.xlsx` files containing False Positive Rate (FPR) and True Positive Rate (TPR).
- **Confusion Matrices**: Printed to the console and can be further analyzed as needed.


## License

This project is licensed under the [MIT License](LICENSE).

## Contact

For any inquiries or support, please contact **Chen Wei** at [begin121@163.com](mailto:begin121@163.com).

---
