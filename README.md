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

### 3. Optimization Technique
We use the SLSQP (Sequential Least Squares Programming) algorithm from SciPy's `minimize` function because:
- It can handle both bounds and constraints
- It's efficient for medium-sized problems
- It works well with the sum of squares objective

### 4. Computational Complexity
- The problem has 30 + (30 × 8) = 270 variables
- The objective function has O(n²) complexity where n is the number of days (30)
- The optimization typically converges in 100-1000 iterations
- Memory usage is moderate, primarily for storing the pricing matrix

### 5. Validation and Error Reporting
- We calculate the mean absolute percentage error (MAPE) between the calculated and target prices
- We visualize the base rates and discount curves
- We save the results to CSV files for further analysis

## How to Run

1. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

2. Place your `pricing_matrix_30x30.csv` file in the project directory

3. Run the script:
   ```
   python pricing_decomposition.py
   ```

4. The script will:
   - Load the pricing matrix
   - Run the optimization
   - Save the results to `base_rates.csv` and `discounts.csv`
   - Generate a visualization in `pricing_decomposition_results.png`

## Output
- `base_rates.csv`: 30 base rates (one per day)
- `discounts.csv`: 30×8 matrix of discount percentages (rows=days, columns=cut-offs)
- `pricing_decomposition_results.png`: Visualization of the results

## Dependencies
- Python 3.7+
- numpy
- pandas
- scipy
- matplotlib
- seaborn
