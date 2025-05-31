# Pricing Matrix Decomposition

This project solves the pricing matrix decomposition problem where we need to determine:
1. A vector of 30 nightly base rates

## Problem Description
Given a 30×30 upper-triangular matrix of stay prices where:
- Rows 1-30 represent check-in days within a month (Day 1 to Day 30)
- Columns 1-30 represent lengths of stay in nights (1 to 30)
- Cell (i,j) holds the price for a stay beginning on Day i and lasting j nights

We need to find the base rates and discount tiers that minimize the discrepancy between the calculated prices and the target matrix.

## Solution Approach

### 1. Objective Function
We minimize the sum of squared differences between the calculated prices and the target prices. This is appropriate because:
- It penalizes large errors more than small ones
- It's differentiable, which helps with optimization
- It's commonly used in regression problems

### 2. Decision Variables and Constraints
- **Base rates (30 variables)**: One for each day, must be positive
- **Discounts (30 × 8 variables)**: Discount percentages for each day and cut-off
- **Constraints**:
  - Discounts must be between 0 and 1 (0% to 100%)
  - For each day, discounts must be strictly increasing with cut-off length

### 3. Optimization Technique 🛠️
We primarily use the **SLSQP (Sequential Least Squares Programming)** algorithm from SciPy's `minimize` function. 
- It handles bounds and non-linear constraints effectively.
- If SLSQP doesn't yield a satisfactory result (based on an error metric combining MAE and scaled MAPE) or fails, the system automatically falls back to **COBYLA (Constrained Optimization BY Linear Approximation)**, and then to **Nelder-Mead** as a final attempt to find a robust solution.

### 4. Computational Complexity
- The problem has 30 + (30 × 8) = 270 variables
- The objective function has O(n²) complexity where n is the number of days (30)
- The optimization typically converges in 100-1000 iterations
- Memory usage is moderate, primarily for storing the pricing matrix

### 5. Validation and Error Reporting 📉
- **Error Metrics**: The script calculates and reports Mean Absolute Error (MAE), Mean Squared Error (MSE), Mean Absolute Percentage Error (MAPE), Median Absolute Percentage Error (MedAPE), and Maximum Absolute Error.
- **Visualizations**: 
    - `original_prices.png`: Heatmap of the input matrix.
    - `base_rates.png`: Plot of the derived base rates.
    - `discount_curves.png`: Plot of the derived discount tiers.
    - `calculated_prices.png`: Heatmap of the reconstructed price matrix.
    - `error_heatmap.png`: Heatmap of absolute errors between original and calculated prices.
- **CSV Outputs**: Detailed results are saved for further analysis.

## How to Run

1. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

2. Place your `pricing_matrix_30x30.csv` file in the project directory

3. Run the script (use `py` on Windows if `python` is not aliased):
   ```bash
   py pricing_decomposition.py
   ```

4. The script will ✅:
   - Load the `pricing_matrix_30x30.csv`.
   - Run the optimization process, attempting multiple solvers if necessary.
   - Save detailed results to CSV files:
     - `vector.csv`: Derived base rates for each day.
     - `discounts.csv`: Derived discount percentages for each day and length-of-stay tier.
     - `price_comparison.csv`: Side-by-side comparison of original and calculated prices.
   - Generate and save visualizations as PNG files (see list below).

## Output Files 📂

**CSV Files:**
- `vector.csv`: Contains the 30 derived base rates, one for each day.
- `discounts.csv`: A 30x8 matrix showing discount percentages. Rows correspond to check-in days, and columns correspond to predefined length-of-stay cutoffs (2, 3, 4, 5, 6, 7, 14, 28 days).
- `price_comparison.csv`: Lists original prices, calculated/reconstructed prices, and the absolute error for all valid stay combinations.

**Image Files (Visualizations - PNG):**
- `original_prices.png`: A heatmap visualization of the input pricing matrix.
- `base_rates.png`: A line plot showing the trend of the derived base rates over the 30 days.
- `discount_curves.png`: Line plots illustrating the discount structures (percentage vs. length of stay) for each of the 30 days.
- `calculated_prices.png`: A heatmap of the price matrix reconstructed using the optimized base rates and discounts.
- `error_heatmap.png`: A heatmap showing the absolute differences between the original and the reconstructed prices, highlighting areas of larger discrepancies.

## Dependencies
- Python 3.7+
- numpy
- pandas
- scipy
- matplotlib
- seaborn
