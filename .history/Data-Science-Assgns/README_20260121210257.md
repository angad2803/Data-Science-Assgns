# Data Science Assignments - Data Distribution Analysis

## Overview
This repository contains a comprehensive data analysis project focused on NO2 (Nitrogen Dioxide) concentration data. The project implements data transformation and statistical parameter estimation using the Method of Moments approach.

## Project Structure
- `All_Assgns.ipynb` - Main Jupyter notebook containing all analysis steps
- `data.csv` - Source dataset with NO2 measurements (not included in repository)

## Methodology

### Step 1: Data Loading and Transformation

#### 1.1 Parameter Setup
The analysis begins with personalized parameters derived from a roll number:

```
Roll Number (r): 102303993
ar = 0.05 × (r mod 7)
br = 0.3 × ((r mod 5) + 1)
```

These parameters are used to create a unique transformation for each student's analysis.

#### 1.2 Data Loading
- **Data Source**: `data.csv` containing NO2 concentration measurements
- **Encoding**: Latin-1 encoding for robust character handling
- **Error Handling**: Implements robust error handling for malformed lines
- **Column Selection**: Automatically detects 'NO2' column (case-insensitive)
- **Data Cleaning**: 
  - Removes leading/trailing spaces from column names
  - Converts data to numeric format
  - Drops NaN values to ensure clean dataset

#### 1.3 Data Transformation
The core transformation applies a sinusoidal function to the original NO2 data:

**Transformation Equation:**
```
z = x + ar × sin(br × x)
```

Where:
- `x` = Original NO2 concentration value
- `z` = Transformed value
- `ar`, `br` = Roll number-dependent parameters

**Purpose**: This transformation creates a non-linear relationship that tests the ability to estimate distribution parameters from transformed data.

### Step 2: Parameter Estimation using Method of Moments

The Method of Moments is a classical statistical technique for parameter estimation. The implementation follows these steps:

#### 2.1 Mean Calculation (μ)
```
μ = (1/n) × Σ(zi)
```
The sample mean provides the first moment of the distribution.

#### 2.2 Variance and Standard Deviation
```
σ² = (1/(n-1)) × Σ(zi - μ)²
σ = √(σ²)
```
These measure the spread of the transformed data around the mean.

#### 2.3 Lambda Parameter (λ)
The lambda parameter is derived by matching the Gaussian/Normal distribution form:

**Standard Normal Form:**
```
f(z) = c × exp(-λ(z-μ)²)
```

**Relationship to Variance:**
```
λ = 1/(2σ²)
```

This parameter controls the rate of exponential decay in the probability density function.

#### 2.4 Normalization Constant (c)
```
c = 1/(σ × √(2π))
```

This constant ensures the probability density function integrates to 1, maintaining proper probability distribution properties.

## Mathematical Background

### Gaussian Distribution
The project fits data to a Gaussian (Normal) distribution of the form:

```
p(z|μ,λ) = c × exp(-λ(z-μ)²)
```

This can be rewritten in standard form as:
```
p(z|μ,σ) = (1/(σ√(2π))) × exp(-(z-μ)²/(2σ²))
```

### Method of Moments Theory
The Method of Moments estimates parameters by equating:
1. Sample moments (calculated from data)
2. Population moments (theoretical expressions involving parameters)

For a Normal distribution:
- **1st Moment**: E[Z] = μ
- **2nd Central Moment**: Var[Z] = σ²

## Results

### Typical Output Format

#### Transformation Parameters
```
Parameters for Roll Number 102303993:
ar = [calculated value]
br = [calculated value]
```

#### Data Loading Results
```
Found target column: 'NO2'
Data loaded successfully. Valid records: [number of records]
```

#### Transformation Preview
A sample of the first 5 rows showing original (x) and transformed (z) values:

| Index | x (Original) | z (Transformed) |
|-------|--------------|-----------------|
| 0     | value_1      | transformed_1   |
| 1     | value_2      | transformed_2   |
| 2     | value_3      | transformed_3   |
| 3     | value_4      | transformed_4   |
| 4     | value_5      | transformed_5   |

#### Learned Parameters
```
Step 2 Results: Learned Parameters
--------------------------------
Mean (μ)      : [value]
Lambda (λ)    : [value]
Constant (c)  : [value]
```

### Interpretation of Results

1. **Mean (μ)**: Central tendency of the transformed NO2 data
2. **Lambda (λ)**: Precision parameter; higher values indicate tighter clustering around the mean
3. **Constant (c)**: Normalization factor ensuring valid probability distribution

## Technical Implementation Details

### Libraries Used
- **pandas**: Data loading and manipulation
- **numpy**: Numerical computations and transformations

### Key Features
1. **Robust Error Handling**: Gracefully handles missing files, malformed data, and encoding issues
2. **Case-Insensitive Column Detection**: Automatically finds the NO2 column regardless of capitalization
3. **Data Validation**: Ensures numeric conversion and removes invalid entries
4. **Modular Structure**: Clear separation of data loading, transformation, and parameter estimation steps

## How to Run

1. **Prerequisites**:
   ```bash
   pip install pandas numpy
   ```

2. **Data Setup**:
   - Place your `data.csv` file in the same directory as the notebook
   - Ensure the CSV contains a column named 'NO2' (case-insensitive)

3. **Execution**:
   - Open `All_Assgns.ipynb` in Jupyter Notebook or JupyterLab
   - Run cells sequentially from top to bottom
   - Review outputs at each step

## Expected Workflow

1. **Step 1 Execution**:
   - Verify roll number and calculated parameters (ar, br)
   - Confirm successful data loading
   - Inspect transformation preview

2. **Step 2 Execution**:
   - Review learned parameters (μ, λ, c)
   - Validate that parameters are reasonable (no NaN or infinite values)

## Statistical Significance

The estimated parameters allow you to:
- Model the transformed NO2 data as a Gaussian distribution
- Make probabilistic predictions about future NO2 measurements
- Understand the central tendency and variability in the data
- Apply maximum likelihood estimation or Bayesian inference in subsequent analyses

## Future Extensions

Potential enhancements to this project:
1. **Visualization**: Add histograms and fitted distribution curves
2. **Model Validation**: Implement goodness-of-fit tests (Chi-square, Kolmogorov-Smirnov)
3. **Comparison**: Compare Method of Moments with Maximum Likelihood Estimation
4. **Multiple Distributions**: Test fit with other distributions (Exponential, Gamma, etc.)
5. **Time Series Analysis**: Explore temporal patterns in NO2 data

## References

- **Method of Moments**: Classical statistical estimation technique
- **Gaussian Distribution**: Fundamental probability distribution in statistics
- **Environmental Data Analysis**: Application to air quality monitoring (NO2 levels)

## Author
Roll Number: 102303993

## License
This project is for educational purposes.

---

**Note**: Ensure `data.csv` is available before running the notebook. The file should contain air quality measurements with an 'NO2' column.
