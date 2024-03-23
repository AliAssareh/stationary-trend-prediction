# Deep Learning for High-Frequency Cryptocurrency Trend Detection

This repository contains the code and datasets used in the research paper titled "Deep Learning for High-Frequency Cryptocurrency Trend Detection: Incorporating Technical Indicators and A New Approach For Data Stationarity". Our study focuses on developing deep learning models to predict cryptocurrency trends with high accuracy, leveraging technical indicators and a novel approach for data stationarity.

## Table of Contents

- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

## Installation

To set up your environment to run the code, please follow the steps below:

1. Clone the repository to your local machine.
2. Ensure you have Python installed, along with Jupyter Notebook or JupyterLab.
3. Install the required Python packages using the requirements file provided in the repository or manually install the necessary libraries.

## Dataset

The dataset used in this study is derived from the Binance exchange API. It includes both raw and preprocessed data necessary for running the experiments.

- **Location**: The data is stored in the `Data` folder in zip format.
- **Preparation**: Before running the experiments, you must extract the zip files to access the datasets.

## Usage

The core of the research is structured into different .py files located in the `Codes` folder. To replicate our experiments or use the models for your own data, follow these steps:

### Preprocessing

- **File**: `0_Preprocessing.ipynb`
- **Description**: This notebook contains all preprocessing steps required to prepare the data for experiments.

### Experiments

- **File**: `0_SecondOrder.ipynb`
- **Description**: This notebook includes the experiments conducted in our study. Run this notebook to replicate our results.

### Custom Experiments

- **File**: `CNN_model.ipynb`
- **Description**: If you wish to run single experiments or customize the experiments, you can use this notebook. Modify the arguments of the pipeline as needed to fit your experimental setup.

## Contributing

We welcome contributions and improvements to our codebase. Please follow the standard GitHub pull request process to submit your changes. Ensure your code adheres to the existing style to maintain consistency.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
## Citation

If you use the code or data from this repository in your research, please cite our paper as follows:

citation key would be included here once the paper is accepted please keep visiting.
