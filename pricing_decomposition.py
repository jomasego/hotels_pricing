import numpy as np
import pandas as pd
from scipy.optimize import minimize
import pulp
import matplotlib.pyplot as plt
import seaborn as sns

class PricingMatrixDecomposer:
    def __init__(self, matrix_file):
        """
        Initialize the pricing matrix decomposer with the input matrix file.
        
        Args:
            matrix_file (str): Path to the CSV file containing the pricing matrix
        """
        self.matrix = self._load_matrix(matrix_file)
        self.n_days = self.matrix.shape[0]
        self.cutoffs = np.array([2, 3, 4, 5, 6, 7, 14, 28])
        self.n_cutoffs = len(self.cutoffs)
        
    def _load_matrix(self, file_path):
        """Load the pricing matrix from CSV file."""
        # Read CSV and convert all values to numeric, setting non-numeric values to NaN
        df = pd.read_csv(file_path, header=None)
        # Convert all values to numeric, coercing errors to NaN
        df = df.apply(pd.to_numeric, errors='coerce')
        # Convert to numpy array
        return df.values.astype(float)
    
    def _get_discount_tier(self, los):
        """Get the discount tier index for a given length of stay."""
        return np.searchsorted(self.cutoffs, los, side='right')
    
    def calculate_prices(self, base_rates, discounts):
        """
        Calculate the price matrix using the given base rates and discounts.
        
        Args:
            base_rates (np.array): Array of length 30 with base rates for each day
            discounts (np.array): 30x8 array of discount percentages (0-1) for each day and cutoff
            
        Returns:
            np.array: 30x30 price matrix
        """
        n_days = len(base_rates)
        calculated = np.zeros((n_days, n_days))
        
        for d in range(n_days):  # Check-in day
            for k in range(1, n_days - d + 1):  # Length of stay
                total_price = 0
                for i in range(k):  # For each night of the stay
                    day = d + i
                    if day >= n_days:
                        continue
                        
                    # Get the discount tier for this length of stay
                    tier = self._get_discount_tier(k)
                    # Apply the appropriate discount
                    discount = discounts[day, tier] if tier < self.n_cutoffs else 0
                    total_price += base_rates[day] * (1 - discount)
                
                calculated[d, k-1] = total_price
                
        return calculated
    
    def objective_function(self, x):
        """
        Calculate the sum of squared differences between calculated and target prices.
        
        Args:
            x (np.array): Flattened array containing base rates followed by discounts
            
        Returns:
            float: Sum of squared differences
        """
        try:
            # Split x into base rates and discounts
            base_rates = x[:self.n_days]
            discounts = x[self.n_days:].reshape((self.n_days, self.n_cutoffs))
            
            # Ensure discounts are monotonically increasing
            for day in range(self.n_days):
                for i in range(1, self.n_cutoffs):
                    if discounts[day, i] < discounts[day, i-1]:
                        discounts[day, i] = discounts[day, i-1] + 0.01
            
            # Calculate prices with current parameters
            calculated = self.calculate_prices(base_rates, discounts)
            
            # Calculate sum of squared differences (only for valid entries)
            mask = ~np.isnan(self.matrix)
            diff = (calculated[mask] - self.matrix[mask])
            
            # Add a small penalty for very large values to improve stability
            penalty = 0.0
            if np.any(calculated > 1e6):
                penalty = np.sum(calculated[calculated > 1e6] ** 2) * 10
                
            return np.sum(diff ** 2) + penalty
            
        except Exception as e:
            # Return a large value if there's an error
            print(f"Warning in objective function: {str(e)}")
            return 1e20
    
    def solve(self, method='SLSQP', max_iter=1000):
        """
        Solve the optimization problem to find optimal base rates and discounts.
        
        Args:
            method (str): Optimization method to use
            max_iter (int): Maximum number of iterations
            
        Returns:
            dict: Dictionary containing the solution
        """
        # Initial guess (base rates and discounts)
        # Start with reasonable initial values
        x0 = np.ones(self.n_days + self.n_days * self.n_cutoffs) * 100  # Base rates around 100
        
        # Initialize discounts with increasing values
        for day in range(self.n_days):
            base_idx = self.n_days + day * self.n_cutoffs
            for i in range(self.n_cutoffs):
                x0[base_idx + i] = 0.05 + 0.1 * (i / self.n_cutoffs)  # 5% to 15% range
        
        # Bounds
        bounds = [(1, None)] * self.n_days  # Base rates must be positive
        
        # Add bounds for discounts (0.01 <= discount < 0.99 and increasing within each day)
        for _ in range(self.n_days * self.n_cutoffs):
            bounds.append((0.01, 0.99))  # Discounts between 1% and 99%
        
        # Constraints for increasing discounts with a minimum step of 0.01
        constraints = []
        for day in range(self.n_days):
            for i in range(1, self.n_cutoffs):
                def make_constraint(day=day, i=i):
                    def f(x):
                        base_idx = self.n_days + day * self.n_cutoffs
                        return x[base_idx + i] - x[base_idx + i - 1] - 0.01
                    return {'type': 'ineq', 'fun': f}
                constraints.append(make_constraint())
        
        # Add a constraint to keep base rates reasonable (not too high)
        def base_rate_constraint(x):
            return 1000 - np.max(x[:self.n_days])
        
        constraints.append({'type': 'ineq', 'fun': base_rate_constraint})
        
        try:
            # First try with SLSQP
            result = minimize(
                self.objective_function,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': max_iter, 'ftol': 1e-6, 'disp': True}
            )
            
            # If SLSQP fails, try with COBYLA which might be more robust with constraints
            if not result.success and method != 'COBYLA':
                print("SLSQP failed, trying COBYLA...")
                # Convert constraints to COBYLA format
                cobyla_constraints = []
                for c in constraints:
                    if 'fun' in c:
                        cobyla_constraints.append(lambda x, c=c: c['fun'](x))
                
                result = minimize(
                    self.objective_function,
                    x0,
                    method='COBYLA',
                    options={'maxiter': max_iter, 'disp': True},
                    constraints=[{'type': 'ineq', 'fun': f} for f in cobyla_constraints]
                )
                
        except Exception as e:
            print(f"Optimization failed with error: {str(e)}")
            result = type('obj', (object,), {'x': x0, 'success': False, 'message': str(e)})()
        
        # Extract solution
        base_rates = result.x[:self.n_days]
        discounts = result.x[self.n_days:].reshape((self.n_days, self.n_cutoffs))
        
        # Ensure discounts are properly ordered (monotonically increasing)
        for day in range(self.n_days):
            for i in range(1, self.n_cutoffs):
                if discounts[day, i] < discounts[day, i-1]:
                    discounts[day, i] = discounts[day, i-1] + 0.01
        
        # Save base rates to CSV
        base_rates_df = pd.DataFrame({
            'Day': range(1, self.n_days + 1),
            'Base_Rate': base_rates
        })
        base_rates_df.to_csv('vector.csv', index=False, float_format='%.6f')
        
        # Save discounts to CSV
        cutoffs = [2, 3, 4, 5, 6, 7, 14, 28]
        discounts_df = pd.DataFrame(discounts, 
                                  columns=[f'Cutoff_{c}' for c in cutoffs])
        discounts_df.insert(0, 'Day', range(1, self.n_days + 1))
        discounts_df.to_csv('discounts.csv', index=False, float_format='%.6f')
        
        # Calculate final error
        calculated = self.calculate_prices(base_rates, discounts)
        mask = ~np.isnan(self.matrix)
        error = np.mean(np.abs(calculated[mask] - self.matrix[mask]) / self.matrix[mask]) * 100
        
        return {
            'base_rates': base_rates,
            'discounts': discounts,
            'error': error,
            'calculated_matrix': calculated,
            'success': result.success,
            'message': result.message
        }

def main():
    try:
        # Load the pricing matrix
        matrix_file = 'pricing_matrix_30x30.csv'
        print(f"Loading pricing matrix from {matrix_file}...")
        decomposer = PricingMatrixDecomposer(matrix_file)
        
        # Verify the matrix was loaded correctly
        if np.isnan(decomposer.matrix).all():
            raise ValueError("Failed to load valid numeric data from the matrix file.")
            
        print(f"Matrix loaded with shape: {decomposer.matrix.shape}")
        print(f"Number of valid entries: {np.sum(~np.isnan(decomposer.matrix))}")
        
        # Solve the optimization problem
        print("\nStarting optimization (this may take a few minutes)...")
        result = decomposer.solve()
        
        if result['success']:
            print("\nOptimization successful!")
            print(f"Mean absolute percentage error: {result['error']:.2f}%")
            
            # Save results
            np.savetxt('base_rates.csv', result['base_rates'], delimiter=',')
            np.savetxt('discounts.csv', result['discounts'], delimiter=',')
            print("\nResults saved to base_rates.csv and discounts.csv")
            
            # Plot results
            print("Generating visualizations...")
            plt.figure(figsize=(15, 6))
            
            # Plot 1: Base Rates
            plt.subplot(1, 2, 1)
            plt.plot(result['base_rates'])
            plt.title('Base Rates by Day')
            plt.xlabel('Day')
            plt.ylabel('Base Rate')
            plt.grid(True)
            
            # Plot 2: Discounts
            plt.subplot(1, 2, 2)
            for i in range(8):
                plt.plot(result['discounts'][:, i], label=f'Cutoff {decomposer.cutoffs[i]}')
            plt.title('Discounts by Day')
            plt.xlabel('Day')
            plt.ylabel('Discount')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig('pricing_decomposition_results.png')
            print("Visualization saved to pricing_decomposition_results.png")
            
            # Show a sample of the results
            print("\nSample of results:")
            print("\nBase Rates (first 5 days):")
            print(result['base_rates'][:5])
            print("\nDiscounts for first day (8 cutoffs):")
            print(result['discounts'][0])
            
    except FileNotFoundError as e:
        print(f"Error: {e}. Please make sure the file exists and try again.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    finally:
        print("\nOptimization process completed.")

if __name__ == "__main__":
    main()
