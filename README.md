# RobustICA


## 📁 Project Structure

### Core Implementation Files

#### `RICA.py`

The main implementation of the RICA (Robust Independent Component Analysis) algorithm. Contains:

- JAX-based implementation with GPU/Metal acceleration support
- `SignatureComputer` class for computing signature moments
- `ContrastCalculator` class for delta computation and contrast functions  
- `Optimizer` class implementing the RICA optimization algorithm
- Utility functions for error computation and matrix operations

#### `continuity_moduli.py`

Implementation of theoretical analysis classes for testing Theorems 4.3 and 4.6:

- `ContinuityModuli` class for continuity moduli of Theorem 4.3
- `RobustnessModuli` class for the constants of Theorem 4.6

#### `sample_different_S.py`

Signal generation utilities with support for various time series types:

- `SignalGenerator` class for synthetic signal creation
- Support for IID, Ornstein-Uhlenbeck (OU), Moving Average, and ARMA processes
- Signal confounding methods (multiplicative, additive, talking pairs, etc.)

#### `SOBI.py`

Implementation of SOBI (Second-Order Blind Identification) algorithm for comparison with RICA. Adapted from https://github.com/davidrigie/sobi/blob/master/sobi/sobi.py 

### Experimental Scripts

#### `robustness_test.py`

Comprehensive robustness analysis script:

- Multi-algorithm comparison (RICA, FastICA, SOBI)
- Various confounding strength testing
- Statistical analysis and visualization

#### `different_S_43.py`

Extended testing for different signal types and Theorem 4.3 validation.

#### `perturbation_and_delta.py`

Analysis of perturbation effects and delta parameter relationships.

#### `plot_what_confounding_does.py`

Visualization tool for understanding the effects of signal confounding:

- Comparison of original vs. confounded signals
- Delta parameter analysis across confounding levels


### Application Folders

#### `cocktail_party/`

Real-world application of ICA to the cocktail party problem:

- **Audio Files**: `gray.mp3`, `goethe.mp3`, `kordian.mp3` - Source audio signals
- **`cocktail_party.py`** - Main cocktail party separation implementation
- **`robustness_cocktail_party.py`** - Robustness analysis with real audio data
- **`unmix_cocktail_party.py`** - Audio unmixing utilities
- **`generate_unmixed_mp3.py`** - Generate separated audio outputs



#### `robustness/`

Collection of generated robustness analysis plots and PDFs:

- Theorem 4.3 validation plots (with and without masks)
- IC-defect analysis results


### Visualization and Results

#### Generated Plots

- **`delta_vs_confounder_strength.png`** - IID signals analysis
- **`delta_vs_confounder_strengthOU.png`** - Ornstein-Uhlenbeck process analysis  
- **`delta_vs_confounder_strength_iid.png`** - Additional IID analysis

#### `point_cloud.ipynb`

Jupyter notebook for point cloud visualization and analysis of ICA results.

### Configuration

#### `requirements.txt`

Project dependencies:

```txt
numpy
matplotlib
scikit-learn
scipy
statsmodels
signax
jax[cuda12_pip]<0.6.0
jaxlib<0.6.0
```

#### `__pycache__/`

Python bytecode cache directory (auto-generated).

## 🔬 Research Focus

This implementation primarily focuses on:

1. **RICA Algorithm**: A robust version of ICA that uses signature-based contrast functions
2. **Theoretical Analysis**: Implementation and validation of Theorems 4.3 and 4.6 regarding IC-defect bounds
3. **Robustness Studies**: Comprehensive analysis of algorithm performance under various confounding conditions
4. **Real-world Applications**: Cocktail party problem and music source separation
5. **Comparative Analysis**: RICA vs. FastICA vs. SOBI performance comparisons

## 🚀 Getting Started

1. Install dependencies: `pip install -r requirements.txt`
2. Run basic tests: `python test_43.py`
3. Explore applications: Navigate to `cocktail_party/` or `music_mixing/` folders
4. Generate visualizations: Use plotting scripts like `plot_what_confounding_does.py`

## 📊 Key Features

- **JAX Integration**: High-performance computing with GPU/Metal acceleration
- **Multiple Signal Types**: Support for IID, OU, and ARMA processes
- **Identifiability Criteria checks**: Allows for testing of mean stationarity and the diagonal third-order identifiability condition.
- **Comprehensive Testing**: Extensive robustness and theoretical validation

## 📈 Experimental Results

The folder contains extensive experimental validation including:

- Theorem 4.3 absolute and relative error bound validation
- IC-defect parameter analysis
- Multi-algorithm performance comparisons
- Real-world audio separation quality assessment
- Robustness analysis under various confounding scenarios