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
        """
        Load the pricing matrix from CSV file.
        
        The CSV is expected to have:
        - First column: 'StartDay' with values like 'Day 1', 'Day 2', etc.
        - Remaining columns: 1-30 for length of stay
        - Empty cells should be treated as NaN
        """
        # Read CSV with header and index_col=0 to use first column as index
        df = pd.read_csv(file_path, index_col=0)
        
        # Convert all values to numeric, coercing errors to NaN
        df = df.apply(pd.to_numeric, errors='coerce')
        
        # Convert to numpy array and return
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
            np.array: 30x30 price matrix with NaN for invalid combinations
        """
        n_days = len(base_rates)
        calculated = np.full((n_days, n_days), np.nan)  # Initialize with NaN
        
        for d in range(n_days):  # Check-in day
            max_stay = n_days - d  # Maximum possible stay length from this day
            
            for k in range(1, max_stay + 1):  # Length of stay (1 to max_stay)
                total_price = 0.0
                
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
        Compute the objective function value for the current solution vector.
        """
        try:
            # Split the solution vector into base rates and discounts
            base_rates = x[:self.n_days]
            discounts = x[self.n_days:].reshape((self.n_days, self.n_cutoffs))
            
            # Ensure base rates are positive
            if np.any(base_rates <= 0):
                return 1e15  # Heavy penalty for invalid base rates
            
            # Clip discounts to valid range [0.01, 0.99] to prevent division by zero or negative discounts
            discounts = np.clip(discounts, 0.01, 0.99)
            
            # Calculate the prices based on the current parameters
            calculated_matrix = self.calculate_prices(base_rates, discounts)
            
            # Compute the squared error between the calculated and target matrices
            # Only for cells that have valid values in the target matrix
            error = calculated_matrix - self.matrix
            valid_mask = ~np.isnan(self.matrix)
            
            if np.sum(valid_mask) == 0:
                return 1e15  # Return a large number if there are no valid entries
                
            # Check if calculated matrix contains NaN values where it shouldn't
            if np.any(np.isnan(calculated_matrix[valid_mask])):
                return 1e15  # Heavy penalty for NaN values
            
            # Use weighted squared error to emphasize lower prices more
            weights = 1.0 / (np.maximum(self.matrix[valid_mask], 1.0) ** 0.5)  # Use sqrt weighting for better scaling
            weighted_sq_error = np.sum(weights * (error[valid_mask]) ** 2) / np.sum(valid_mask)
            
            # Add penalty for non-monotonic discounts within each day
            penalty = 0.0
            for d in range(self.n_days):
                for i in range(1, self.n_cutoffs):
                    if discounts[d, i] < discounts[d, i-1]:
                        penalty += 1e6 * (discounts[d, i-1] - discounts[d, i]) ** 2
            
            # Add L2 regularization to prevent extreme values
            regularization = 1e-4 * np.sum(base_rates ** 2) / self.n_days
            
            # Penalize extreme discounts that are close to boundaries
            boundary_penalty = 1e4 * np.sum((discounts < 0.05) | (discounts > 0.95))
            
            # Return the weighted squared error plus any penalties
            result = weighted_sq_error + penalty + regularization + boundary_penalty
            
            # Check for numerical issues
            if np.isnan(result) or np.isinf(result):
                return 1e15
                
            return result
            
        except Exception as e:
            print(f"Error in objective function: {str(e)}")
            return 1e15  # Return a large number if there was an error
    
    def solve(self, method='SLSQP', max_iter=1000):
        """
        Solve the optimization problem to find the best base rates and discounts.
        """
        print(f"\nStarting optimization with method: {method} (max {max_iter} iterations)...")

        valid_prices = self.matrix[~np.isnan(self.matrix)]
        if len(valid_prices) == 0:
            raise ValueError("No valid prices found in the matrix")

        min_price = np.min(valid_prices)
        max_price = np.max(valid_prices)
        avg_price = np.mean(valid_prices)
        median_price = np.median(valid_prices)
        print(f"Price statistics - Min: {min_price:.2f}, Max: {max_price:.2f}, Mean: {avg_price:.2f}, Median: {median_price:.2f}")

        base_rates_guess = np.zeros(self.n_days)
        for d in range(self.n_days):
            if d < self.matrix.shape[1] and not np.isnan(self.matrix[d, 0]):
                base_rates_guess[d] = self.matrix[d, 0]
            else:
                day_prices = self.matrix[d, :]
                if np.any(~np.isnan(day_prices)):
                    base_rates_guess[d] = np.nanmean(day_prices) / 2
                else:
                    base_rates_guess[d] = median_price / 2
        base_rates_guess = np.clip(base_rates_guess, min_price * 0.5, max_price)
        bad_indices = np.where(np.isnan(base_rates_guess) | (base_rates_guess <= 0))[0]
        if len(bad_indices) > 0:
            base_rates_guess[bad_indices] = median_price / 2

        discounts_guess = np.zeros((self.n_days, self.n_cutoffs))
        for d in range(self.n_days):
            for i in range(self.n_cutoffs):
                discounts_guess[d, i] = 0.10 + 0.04 * i + 0.01 * d
        discounts_guess = np.clip(discounts_guess, 0.05, 0.60)
        for d in range(self.n_days):
            for i in range(1, self.n_cutoffs):
                if discounts_guess[d, i] <= discounts_guess[d, i-1]:
                    discounts_guess[d, i] = discounts_guess[d, i-1] + 0.01

        np.random.seed(42)
        base_rates_guess *= (0.95 + 0.1 * np.random.rand(self.n_days))
        discounts_guess *= (0.98 + 0.04 * np.random.rand(*discounts_guess.shape))
        x0 = np.concatenate([base_rates_guess, discounts_guess.flatten()])

        bounds = []
        for i in range(self.n_days):
            bounds.append((max(1.0, min_price * 0.25), max_price * 1.5))
        for i in range(self.n_days * self.n_cutoffs):
            bounds.append((0.01, 0.9))

        constraints = []
        for d_idx in range(self.n_days):
            for c_idx in range(1, self.n_cutoffs):
                def const_fun(x, d=d_idx, i=c_idx):
                    disc_reshaped = x[self.n_days:].reshape((self.n_days, self.n_cutoffs))
                    return disc_reshaped[d, i] - disc_reshaped[d, i-1] - 0.001
                constraints.append({'type': 'ineq', 'fun': const_fun})
        print(f"Setup {len(constraints)} constraints for {len(bounds)} variables")

        def evaluate_solution(res):
            if not res.success:
                return float('inf')
            try:
                br = res.x[:self.n_days]
                disc = res.x[self.n_days:].reshape((self.n_days, self.n_cutoffs))
                calc = self.calculate_prices(br, disc)
                errs = calc - self.matrix
                valid_m = ~np.isnan(self.matrix)
                abs_errs = np.abs(errs[valid_m])
                rel_errs = abs_errs / np.maximum(self.matrix[valid_m], 1.0)
                current_mae = np.mean(abs_errs)
                current_mape = np.mean(rel_errs) * 100
                return current_mae + current_mape / 10
            except Exception as eval_e:
                print(f"Error evaluating solution: {eval_e}")
                return float('inf')

        best_result_obj = None
        best_error_val = float('inf')

        optimizers_to_try = [
            {'method': method, 'options': {'maxiter': max_iter, 'disp': True, 'ftol': 1e-6}},
            {'method': 'COBYLA', 'options': {'maxiter': 2000, 'rhobeg': min(50.0, avg_price/10), 'disp': True}},
            {'method': 'Nelder-Mead', 'options': {'maxiter': 5000, 'disp': True}}
        ]

        for opt_config in optimizers_to_try:
            current_method = opt_config['method']
            current_options = opt_config['options']
            # Skip primary if it's already tried or if error is too high for subsequent ones
            if best_result_obj is not None and current_method == method: # Already tried primary
                 if not best_result_obj.success or best_error_val > 1000 and current_method == 'COBYLA':
                    pass # Allow COBYLA if primary failed badly
                 elif best_error_val > 500 and current_method == 'Nelder-Mead':
                    pass # Allow Nelder-Mead if others were not good enough
                 elif current_method != method: # Only proceed if it's a fallback
                    continue
            
            # For SLSQP, pass constraints and bounds. COBYLA and Nelder-Mead might not use all.
            minimize_args = {'fun': self.objective_function, 'x0': x0, 'method': current_method, 'options': current_options}
            if current_method == 'SLSQP':
                minimize_args['bounds'] = bounds
                minimize_args['constraints'] = constraints
            elif current_method == 'COBYLA': # COBYLA uses constraints but not bounds in the same way
                 minimize_args['constraints'] = constraints

            print(f"\nOptimizing with {current_method}...")
            try:
                current_result = minimize(**minimize_args)
                current_error = evaluate_solution(current_result)
                print(f"{current_method} result - Success: {current_result.success}, Error metric: {current_error:.2f}, Message: {getattr(current_result, 'message', 'N/A')}")

                if current_error < best_error_val:
                    best_result_obj = current_result
                    best_error_val = current_error
                
                # Break if a good enough solution is found (e.g. primary succeeded well)
                if best_result_obj and best_result_obj.success and best_error_val < 500 and current_method == method:
                    break 
            except Exception as opt_e:
                print(f"Optimization with {current_method} failed: {opt_e}")
                import traceback
                traceback.print_exc()

        if best_result_obj is None:
            from types import SimpleNamespace
            best_result_obj = SimpleNamespace()
            best_result_obj.success = False
            best_result_obj.message = "All optimization methods failed or were skipped."
            best_result_obj.x = x0 # Fallback to initial guess

        final_base_rates = best_result_obj.x[:self.n_days]
        final_discounts = best_result_obj.x[self.n_days:].reshape((self.n_days, self.n_cutoffs))

        print("\nPost-processing solution...")
        final_base_rates = np.maximum(1.0, final_base_rates)
        for d_idx in range(self.n_days):
            final_discounts[d_idx] = np.clip(final_discounts[d_idx], 0.01, 0.99)
            for c_idx in range(1, self.n_cutoffs):
                if final_discounts[d_idx, c_idx] <= final_discounts[d_idx, c_idx-1]:
                    final_discounts[d_idx, c_idx] = min(0.99, final_discounts[d_idx, c_idx-1] + 0.001)

        final_calculated_prices = self.calculate_prices(final_base_rates, final_discounts)

        final_errors = final_calculated_prices - self.matrix
        final_valid_mask = ~np.isnan(self.matrix)
        final_abs_errors = np.abs(final_errors[final_valid_mask])
        
        if len(final_abs_errors) == 0: # Handle case with no valid prices for error calculation
            mae, mse, mape, median_ape, max_err = float('nan'), float('nan'), float('nan'), float('nan'), float('nan')
        else:
            mse = np.mean(np.square(final_abs_errors))
            mae = np.mean(final_abs_errors)
            final_rel_errors = final_abs_errors / np.maximum(1.0, self.matrix[final_valid_mask])
            mape = 100 * np.mean(final_rel_errors)
            median_ape = 100 * np.median(final_rel_errors)
            max_err = np.max(final_abs_errors)

        print(f"\nFinal Error Metrics:")
        print(f"  Mean Absolute Error: ${mae:.2f}")
        print(f"  Mean Squared Error: {mse:.2f}")
        print(f"  Mean Absolute Percentage Error: {mape:.2f}%")
        print(f"  Median Absolute Percentage Error: {median_ape:.2f}%")
        print(f"  Max Absolute Error: ${max_err:.2f}")

        try:
            print("\nSaving results to CSV files...")
            pd.DataFrame({'Day': range(1, self.n_days + 1), 'Base_Rate': final_base_rates}).to_csv('vector.csv', index=False)
            print(f"Base rates saved to vector.csv")

            discount_df_final = pd.DataFrame(final_discounts)
            discount_df_final.index.name = 'Day'
            discount_df_final.columns = [f'Tier_{j+1}' for j in range(self.n_cutoffs)]
            discount_df_final.to_csv('discounts.csv')
            print(f"Discount tiers saved to discounts.csv")

            pd.DataFrame({'Original': self.matrix.flatten(), 'Calculated': final_calculated_prices.flatten()}).to_csv('price_comparison.csv', index=False)
            print(f"Price comparison saved to price_comparison.csv")
        except Exception as csv_e:
            print(f"Error saving results to CSV: {csv_e}")

        try:
            self.plot_results(final_base_rates, final_discounts, final_calculated_prices)
        except Exception as plot_e:
            print(f"Error generating visualizations: {plot_e}")

        return {
            'success': best_result_obj.success,
            'message': getattr(best_result_obj, 'message', 'N/A'),
            'base_rates': final_base_rates,
            'discounts': final_discounts,
            'calculated': final_calculated_prices,
            'error_metrics': {
                'mae': mae, 'mse': mse, 'mape': mape,
                'median_ape': median_ape, 'max_error': max_err
            }
        }

    def plot_results(self, base_rates, discounts, calculated):
        """Plot the results of the optimization."""
        # Plot base rates
        plt.figure(figsize=(12, 6))
        plt.plot(base_rates)
        plt.xlabel('Day')
        plt.ylabel('Base Rate')
        plt.title('Base Rates')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('base_rates.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Plot discounts
        plt.figure(figsize=(12, 6))
        for d in range(self.n_days):
            plt.plot(discounts[d], label=f'Day {d+1}')
        plt.xlabel('Length of Stay')
        plt.ylabel('Discount Factor')
        plt.title('Discount Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('discount_curves.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Plot calculated prices
        plt.figure(figsize=(12, 10))
        sns.heatmap(calculated, cmap='viridis', mask=np.isnan(self.matrix))
        plt.title('Calculated Price Matrix')
        plt.tight_layout()
        plt.savefig('calculated_prices.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Plot error heatmap
        error = np.abs(calculated - self.matrix)
        plt.figure(figsize=(12, 10))
        sns.heatmap(error, cmap='Reds', mask=np.isnan(self.matrix), vmin=0, vmax=np.nanmax(error)/2)
        plt.title('Absolute Error Heatmap')
        plt.tight_layout()
        plt.savefig('error_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    # Load the pricing matrix
    print("Loading pricing matrix from pricing_matrix_30x30.csv...")
    try:
        decomposer = PricingMatrixDecomposer("pricing_matrix_30x30.csv")
        print(f"Matrix shape: {decomposer.matrix.shape}")
        print(f"Number of valid entries: {np.sum(~np.isnan(decomposer.matrix))}")
        
        # Plot the original price matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(decomposer.matrix, cmap='viridis', 
                    mask=np.isnan(decomposer.matrix))
        plt.title('Original Price Matrix')
        plt.tight_layout()
        plt.savefig('original_prices.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Error loading matrix: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    # Solve the optimization problem
    print("\nStarting optimization (this may take a few minutes)...")
    try:
        result = decomposer.solve(method='SLSQP', max_iter=1000) # solve now handles plotting
        
        # Print detailed results
        print("\n" + "="*50)
        print("OPTIMIZATION RESULTS")
        print("="*50)
        
        print(f"\nOptimization {'succeeded' if result['success'] else 'failed'}")
        if 'message' in result and result['message'] and result['message'] != 'N/A':
            print(f"Message: {result['message']}")
        
        print("\nPerformance Metrics from optimization:")
        error_metrics = result.get('error_metrics', {})
        print(f"- Mean Absolute Error (MAE): {error_metrics.get('mae', float('nan')):.2f}")
        print(f"- Mean Squared Error (MSE): {error_metrics.get('mse', float('nan')):.2f}")
        print(f"- Mean Absolute Percentage Error (MAPE): {error_metrics.get('mape', float('nan')):.2f}%")
        print(f"- Median Absolute Percentage Error (MedAPE): {error_metrics.get('median_ape', float('nan')):.2f}%")
        print(f"- Maximum Absolute Error: {error_metrics.get('max_error', float('nan')):.2f}")
            
    except Exception as e:
        print(f"\nError during optimization: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*50)
    print("Optimization process completed.")
    print("Results have been saved to:")
    print("- vector.csv: Base rates for each day")
    print("- discounts.csv: Discount factors for each day and length of stay")
    print("- price_comparison.csv: Original vs Calculated prices")
    print("- original_prices.png: Heatmap of the input matrix")
    print("- base_rates.png: Plot of the derived base rates")
    print("- discount_curves.png: Plot of the derived discount tiers")
    print("- calculated_prices.png: Heatmap of the prices reconstructed from base rates and discounts")
    print("- error_heatmap.png: Heatmap of the absolute differences between original and calculated prices")

if __name__ == "__main__":
    main()
