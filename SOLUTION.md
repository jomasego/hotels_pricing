# Pricing Matrix Decomposition - Solution

## Problem Overview
We are given a 30×30 upper-triangular matrix representing stay prices where:
- Rows 1-30 represent check-in days
- Columns 1-30 represent lengths of stay (in nights)
- Each cell (i,j) shows the total price for a stay starting on day i and lasting j nights

## Solution Approach

### 1. Mathematical Formulation

#### Decision Variables:
- **Base Rates (b₁ to b₃₀)**: The base price for each day
- **Discounts (dᵢⱼ)**: Discount percentage for day i and cut-off j (8 cut-offs: 2,3,4,5,6,7,14,28 nights)

#### Objective Function:
Minimize the sum of squared differences between calculated and target prices:
```
minimize Σ(calculated_price(i,j) - target_price(i,j))²
```

#### Constraints:
1. Base rates > 0
2. 0.01 ≤ dᵢⱼ ≤ 0.99 (1% to 99% discount)
3. dᵢⵢ < dᵢⱼ for j > i (monotonically increasing discounts)

### 2. Optimization Methodology ⚙️

#### Algorithm:
- **Primary Optimizer**: Sequential Least Squares Programming (SLSQP).
- **Fallback Optimizers**: 
    1. Constrained Optimization BY Linear Approximation (COBYLA) - Used if SLSQP fails or the solution error metric is too high (e.g., > 500).
    2. Nelder-Mead - Used as a final attempt if COBYLA also fails or its solution is unsatisfactory.
- **Implementation**: Python's `scipy.optimize.minimize`.

#### Key Features:
1. **Robust Initialization** ✨:
   - Base rates initialized based on matrix statistics (e.g., 75% of the mean of valid prices), with added noise for diversity.
   - Discounts initialized to a small percentage (e.g., 5%) with slight random variations, ensuring initial monotonicity.
   - Bounds: Base rates (min $1 or 25% of min price, max 150% of max price), Discounts (1% to 90%).

2. **Constraint Handling** ⚖️:
   - Hard constraints for variable bounds (base rates and discounts).
   - Inequality constraints to enforce strict monotonicity on discount tiers (e.g., `d_k+1 >= d_k + 0.001`).
   - Penalty terms within the objective function for violations of soft constraints and to guide the solution towards desirable properties.

3. **Error Handling & Fallback Logic** 🛡️:
   - Multi-stage optimization: Tries SLSQP first. If it fails or the error metric (combined MAE and scaled MAPE) is above a threshold (e.g., 500), it falls back to COBYLA. If COBYLA also results in a high error or fails, it attempts Nelder-Mead.
   - Solution evaluation helper function to assess quality.
   - Graceful degradation with meaningful error messages and detailed logging.
   - Input validation and data integrity checks during matrix loading.

### 3. Implementation Details

#### Data Flow:
1. **Input**: `pricing_matrix_30x30.csv`
2. **Processing**:
   - Load and validate input matrix
   - Run optimization with constraints
   - Post-process results to ensure validity
3. **Output** 📊:
   - `vector.csv`: Calculated base rates for each day.
   - `discounts.csv`: Calculated discount percentages for each day and cut-off tier.
   - `price_comparison.csv`: Original prices vs. calculated prices for each valid stay.
   - `original_prices.png`: Heatmap of the input pricing matrix.
   - `base_rates.png`: Plot of the derived base rates over the 30 days.
   - `discount_curves.png`: Plot of the derived discount tiers for each day.
   - `calculated_prices.png`: Heatmap of the reconstructed price matrix using the derived base rates and discounts.
   - `error_heatmap.png`: Heatmap of the absolute differences between original and calculated prices.
   - Console output with optimization progress, final metrics, and status messages.

#### Key Functions:
- `_load_matrix()`: Data loading and validation from CSV.
- `calculate_prices()`: Computes the price matrix from given base rates and discounts.
- `objective_function()`: Calculates the weighted sum of squared errors plus penalties.
- `solve()`: Main optimization routine orchestrating initialization, optimization attempts with fallbacks, post-processing, result saving, and plotting.
- `plot_results()`: Generates and saves all visualizations.

### 4. Results and Validation

#### Output Files:
1. **vector.csv**
   ```
   Day,Base_Rate
   1,123.456789
   2,124.567890
   ...
   30,120.987654
   ```

2. **discounts.csv**
   ```
   Day,Cutoff_2,Cutoff_3,Cutoff_4,Cutoff_5,Cutoff_6,Cutoff_7,Cutoff_14,Cutoff_28
   1,0.050000,0.100000,0.150000,0.200000,0.250000,0.300000,0.350000,0.400000
   ...
   30,0.052000,0.102000,0.152000,0.202000,0.252000,0.302000,0.352000,0.402000
   ```

#### Performance Metrics (Example from a successful SLSQP run):
- Mean Absolute Error (MAE): $69.67
- Mean Squared Error (MSE): 25973.70
- Mean Absolute Percentage Error (MAPE): 7.24%
- Median Absolute Percentage Error (MedAPE): 5.55%
- Maximum Absolute Error: $1349.75
- Optimization Status: Succeeded (SLSQP)
- Iterations: 41
- Function Evaluations: 11261

### 5. Usage

#### Dependencies:
```
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

#### Running the Solution:
```bash
# Install dependencies
pip install -r requirements.txt

# Run the script (use `py` on Windows if `python` is not aliased correctly)
py pricing_decomposition.py
```

### 6. Limitations and Future Work

#### Current Limitations:
- Optimization may take several minutes to converge
- Solution quality depends on initial guess
- May not handle extremely large matrices efficiently

#### Potential Improvements:
1. Parallel processing for faster optimization
2. More sophisticated initialization strategies
3. Additional validation and error handling
4. Support for different optimization objectives

### 7. Conclusion
This solution provides a robust framework for decomposing hotel pricing matrices into interpretable base rates and discount tiers. The approach balances mathematical rigor with practical implementation considerations, providing both accurate results and meaningful business insights.
