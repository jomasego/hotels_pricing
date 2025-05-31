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

### 2. Optimization Methodology

#### Algorithm:
- Primary: Sequential Least Squares Programming (SLSQP)
- Fallback: Constrained Optimization BY Linear Approximation (COBYLA)
- Implementation: Python's `scipy.optimize.minimize`

#### Key Features:
1. **Robust Initialization**: 
   - Base rates initialized around 100
   - Discounts initialized in 5-15% range, increasing with cut-off

2. **Constraint Handling**:
   - Hard constraints on variable bounds
   - Soft constraints for monotonic discounts with minimum 1% steps
   - Penalty terms for constraint violations

3. **Error Handling**:
   - Fallback to COBYLA if SLSQP fails
   - Graceful degradation with meaningful error messages
   - Input validation and data integrity checks

### 3. Implementation Details

#### Data Flow:
1. **Input**: `pricing_matrix_30x30.csv`
2. **Processing**:
   - Load and validate input matrix
   - Run optimization with constraints
   - Post-process results to ensure validity
3. **Output**:
   - `vector.csv`: Base rates for each day
   - `discounts.csv`: Discount percentages for each day and cut-off
   - Console output with optimization metrics

#### Key Functions:
- `calculate_prices()`: Compute prices from base rates and discounts
- `objective_function()`: Calculate sum of squared errors
- `solve()`: Main optimization routine
- `_load_matrix()`: Data loading and validation

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

#### Performance Metrics:
- Mean Absolute Percentage Error (MAPE): [value]%
- Optimization Status: [Success/Failed]
- Iterations: [number]
- Function Evaluations: [number]

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

# Run the script
python pricing_decomposition.py
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
