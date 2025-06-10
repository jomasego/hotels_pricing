import numpy as np
import pandas as pd
from scipy.optimize import minimize
import pulp
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple # Added for type hints, can be np.ndarray directly too

def calculate_appe(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    k: float = 3.0, 
    w_neg: float = 2.0, 
    w_pos: float = 1.0
) -> float:
    """
    Calculates the Asymmetric Powered Percentage Error (APPE).

    This loss function is designed to heavily penalize large percentage errors,
    with a more significant penalty for overpricing (predicted price > actual price)
    compared to underpricing.

    Parameters:
    -----------
    y_true : np.ndarray
        An array of true target values (e.g., actual hotel prices).
        All values in y_true are expected to be positive.
    y_pred : np.ndarray
        An array of predicted values from the model, corresponding to y_true.
    k : float, optional
        The exponent (power) to which the absolute percentage error is raised.
        Default: 3.0.
    w_neg : float, optional
        The weight applied to negative percentage errors (instances of overpricing,
        i.e., y_pred > y_true). Default: 2.0.
    w_pos : float, optional
        The weight applied to positive or zero percentage errors (instances of
        underpricing or perfect prediction, i.e., y_pred <= y_true). Default: 1.0.

    Return Value:
    -------------
    float
        A single float value representing the mean APPE.

    Assumptions:
    ------------
    - All elements in y_true must be positive for a meaningful percentage error
      calculation in this pricing context.

    Raises:
    -------
    TypeError
        If y_true or y_pred are not NumPy arrays.
    ValueError
        If y_true and y_pred do not have the same shape.
        If any element in y_true is not positive.

    Example Usage:
    --------------
    >>> actual_prices = np.array([100, 150, 200, 250])
    >>> predicted_prices = np.array([90, 160, 220, 240]) # 2 underprice, 2 overprice
    >>> appe_loss = calculate_appe(actual_prices, predicted_prices, k=3, w_neg=2, w_pos=1)
    >>> print(f"APPE Loss: {appe_loss}")
    # Expected output will depend on precise calculation, but demonstrates usage.
    # For the example:
    # PE1 = (100-90)/100 = 0.1 (underprice) -> cost1 = 1.0 * (0.1^3) = 0.001
    # PE2 = (150-160)/150 = -0.0666 (overprice) -> cost2 = 2.0 * (|-0.0666|^3) approx 2.0 * 0.000296 = 0.000592
    # PE3 = (200-220)/200 = -0.1 (overprice) -> cost3 = 2.0 * (|-0.1|^3) = 2.0 * 0.001 = 0.002
    # PE4 = (250-240)/250 = 0.04 (underprice) -> cost4 = 1.0 * (0.04^3) = 0.000064
    # Mean APPE = (0.001 + 0.000592 + 0.002 + 0.000064) / 4 = 0.003656 / 4 = 0.000914
    """
    if not isinstance(y_true, np.ndarray) or not isinstance(y_pred, np.ndarray):
        raise TypeError("y_true and y_pred must be NumPy arrays.")

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")

    if not np.all(y_true > 0):
        raise ValueError("All elements in y_true must be positive for APPE calculation.")

    # Calculate Percentage Error (PE)
    # PE_i = (y_true_i - y_pred_i) / y_true_i
    # If PE_i < 0, it means y_pred_i > y_true_i (overpricing)
    # If PE_i > 0, it means y_pred_i < y_true_i (underpricing)
    percentage_errors = (y_true - y_pred) / y_true
    
    abs_percentage_errors_powered = np.abs(percentage_errors)**k
    
    # Apply weights:
    # Create a weights array based on the sign of percentage_errors
    # Where PE < 0 (overpricing), use w_neg
    # Where PE >= 0 (underpricing/perfect), use w_pos
    weights = np.where(percentage_errors < 0, w_neg, w_pos)
    
    costs = weights * abs_percentage_errors_powered
    
    mean_appe = np.mean(costs)
    
    return mean_appe


class PricingMatrixDecomposer:
    def __init__(self, matrix_input):
        """
        Initialize the pricing matrix decomposer with the input matrix.
        
        Args:
            matrix_input (str, np.ndarray, pd.DataFrame): 
                Path to the CSV file containing the pricing matrix,
                or a pre-loaded NumPy array, or a Pandas DataFrame.
        """
        if isinstance(matrix_input, str):
            self.matrix = self._load_matrix(matrix_input)
        elif isinstance(matrix_input, np.ndarray):
            self.matrix = matrix_input.astype(float) # Ensure float type
        elif isinstance(matrix_input, pd.DataFrame):
            self.matrix = matrix_input.values.astype(float) # Ensure float type
        else:
            raise TypeError(
                "matrix_input must be a file path (str), NumPy array, or Pandas DataFrame. "
                f"Got {type(matrix_input)}"
            )
        
        if self.matrix.ndim != 2:
            raise ValueError(
                f"Input matrix must be 2-dimensional. Received shape: {self.matrix.shape}"
            )
        
        self.n_days = self.matrix.shape[0]
        if self.n_days == 0:
            raise ValueError(
                "Input matrix must not be empty (0 rows/n_days). Received shape: "
                f"{self.matrix.shape}"
            )
        
        if self.matrix.shape[1] == 0:
             raise ValueError(
                "Input matrix must have at least one column (max_los > 0). Received shape: "
                f"{self.matrix.shape}"
            )

        self.cutoffs = np.array([2, 3, 4, 5, 6, 7, 14, 28]) # LOS cutoffs for discount tiers
        self.n_cutoffs = len(self.cutoffs)
        self.active_objective_type = None # To be set by solve method
        self.objective_filename_suffix = "" # To be set by solve method
        self.appe_params = None # For APPE specific parameters
        self.active_objective_config = None # To store the full config
        
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
        Conditionally calculates error based on self.active_objective_type.
        """
        if np.any(np.isnan(x)) or np.any(np.isinf(x)):
            print(f"DEBUG ({self.active_objective_type}): Input x contains NaN/Inf. Returning 1e15.")
            return 1e15 # Penalty for invalid input x

        try:
            base_rates = x[:self.n_days]
            discounts = x[self.n_days:].reshape((self.n_days, self.n_cutoffs))

            if np.any(base_rates <= 0):
                print(f"DEBUG ({self.active_objective_type}): base_rates <= 0. Returning 1e15.")
                return 1e15
            discounts = np.clip(discounts, 0.01, 0.99)
            calculated_matrix = self.calculate_prices(base_rates, discounts)
            error_values = calculated_matrix - self.matrix
            valid_mask = ~np.isnan(self.matrix)

            if np.sum(valid_mask) == 0:
                print(f"DEBUG ({self.active_objective_type}): No valid entries in mask. Returning 1e15.")
                return 1e15
            if np.any(np.isnan(calculated_matrix[valid_mask])) or np.any(np.isinf(calculated_matrix[valid_mask])):
                print(f"DEBUG ({self.active_objective_type}): calculated_matrix contains NaN/Inf. Returning 1e15.")
                return 1e15

            core_error = 0
            if self.active_objective_type == 'MSPE':
                original_prices_for_pe = np.maximum(self.matrix[valid_mask], 1.0)
                percentage_error = error_values[valid_mask] / original_prices_for_pe
                core_error = np.mean(percentage_error ** 2) # MSPE

            elif self.active_objective_type == 'WMSE':
                weights = 1.0 / (np.maximum(self.matrix[valid_mask], 1.0) ** 0.5)
                core_error = np.sum(weights * (error_values[valid_mask]) ** 2) / np.sum(valid_mask) # Weighted MSE

            elif self.active_objective_type == 'APPE':
                if self.appe_params:
                    core_error = calculate_appe(
                        self.matrix[valid_mask],
                        calculated_matrix[valid_mask],
                        k=self.appe_params.get('k', 3.0),
                        w_neg=self.appe_params.get('w_neg', 2.0),
                        w_pos=self.appe_params.get('w_pos', 1.0)
                    )
                else: # Fallback to defaults
                    core_error = calculate_appe(
                        self.matrix[valid_mask],
                        calculated_matrix[valid_mask]
                    )
            else:
                raise ValueError(f"Unknown objective type: {self.active_objective_type}")

            # Default penalty coefficients
            default_penalty_coeffs = {
                'non_monotonic': 100.0,
                'regularization': 1e-8, # A small default, can be overridden
                'boundary': 1000.0      # A moderate default, can be overridden
            }

            # Get objective-specific penalty coefficients, or use defaults from active_objective_config
            # If 'penalty_coeffs' is not in active_objective_config, or a specific key is missing from it, use the default_penalty_coeffs value.
            config_penalty_coeffs = self.active_objective_config.get('penalty_coeffs', {})
            
            penalty_non_monotonic_coeff = config_penalty_coeffs.get('non_monotonic', default_penalty_coeffs['non_monotonic'])
            regularization_coeff = config_penalty_coeffs.get('regularization', default_penalty_coeffs['regularization'])
            boundary_penalty_coeff = config_penalty_coeffs.get('boundary', default_penalty_coeffs['boundary'])

            # Shared penalty calculations
            monotonicity_penalty_val = 0.0
            for d_idx in range(self.n_days):
                for c_idx in range(1, self.n_cutoffs):
                    if discounts[d_idx, c_idx] < discounts[d_idx, c_idx - 1]:
                        monotonicity_penalty_val += (discounts[d_idx, c_idx - 1] - discounts[d_idx, c_idx]) ** 2
            
            regularization_penalty_val = np.sum(base_rates ** 2)
            
            # Using the more nuanced boundary penalty for both, scaled by coefficient
            # Calculate penalties separately to avoid broadcasting issues with empty arrays
            lower_bound_mask = (discounts < 0.05)
            upper_bound_mask = (discounts > 0.95)
            
            extreme_discount_penalty_val = 0.0
            if np.any(lower_bound_mask):
                extreme_discount_penalty_val += np.sum(lower_bound_mask * (0.05 - discounts)**2)
            if np.any(upper_bound_mask):
                extreme_discount_penalty_val += np.sum(upper_bound_mask * (discounts - 0.95)**2)

            # Apply coefficients
            total_penalty = (
                monotonicity_penalty_val * penalty_non_monotonic_coeff +
                regularization_penalty_val * regularization_coeff +
                extreme_discount_penalty_val * boundary_penalty_coeff
            )
            
            result = core_error + total_penalty

            if np.isnan(result) or np.isinf(result) or result > 1e14:  # Cap very large finite values
                print(f"DEBUG ({self.active_objective_type}): Final result is NaN/Inf or >1e14 ({result=}). Returning 1e15.")
                return 1e15
            return result

        except Exception as e:
            print(f"DEBUG ({self.active_objective_type}): Exception caught: {str(e)}. Returning 1e16.")
            return 1e16 # Distinct penalty for unexpected exceptions
    
    def solve(self, objective_config, method='SLSQP', max_iter=1000):
        """
        Solve the optimization problem to find the best base rates and discounts.
        Uses objective_config to determine error metric and output naming.
        objective_config = {'type': 'MSPE'/'WMSE', 'suffix': '_suffix', 'display_name': 'Display Name'}
        """
        self.active_objective_config = objective_config # Store the whole config
        self.active_objective_type = self.active_objective_config['type']
        self.objective_filename_suffix = self.active_objective_config['suffix']
        objective_display_name = self.active_objective_config['display_name']
        if self.active_objective_type == 'APPE':
            self.appe_params = self.active_objective_config.get('appe_params', {'k': 3.0, 'w_neg': 2.0, 'w_pos': 1.0})
        else:
            self.appe_params = None # Clear if not APPE

        print(f"\n--- Starting Optimization for: {objective_display_name} ({self.active_objective_type}) ---")
        print(f"Primary method: {method} (max {max_iter} iterations)...")

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

                if np.sum(valid_m) == 0:
                    print(f"DEBUG ({self.active_objective_type}): No valid entries in mask. Returning 1e15.")
                    return 1e15
                if np.any(np.isnan(calc[valid_m])) or np.any(np.isinf(calc[valid_m])):
                    print(f"DEBUG ({self.active_objective_type}): calculated_matrix contains NaN/Inf. Returning 1e15.")
                    return 1e15

                if self.active_objective_type == 'MSPE':
                    original_prices_for_pe = np.maximum(self.matrix[valid_m], 1.0)
                    percentage_error = errs[valid_m] / original_prices_for_pe
                    mspe_eval = np.mean(percentage_error ** 2) # MSPE
                    rmspe_eval = np.sqrt(mspe_eval) * 100  # As a percentage
                    return rmspe_eval
                elif self.active_objective_type == 'WMSE':
                    abs_errs = np.abs(errs[valid_m])
                    rel_errs = abs_errs / np.maximum(self.matrix[valid_m], 1.0)
                    current_mae = np.mean(abs_errs)
                    current_mape = np.mean(rel_errs) * 100
                    return current_mae + current_mape / 10 # Original WMSE evaluation metric
                elif self.active_objective_type == 'APPE':
                    if self.appe_params:
                        appe_eval = calculate_appe(
                            self.matrix[valid_m],
                            calc[valid_m],
                            k=self.appe_params.get('k', 3.0),
                            w_neg=self.appe_params.get('w_neg', 2.0),
                            w_pos=self.appe_params.get('w_pos', 1.0)
                        )
                    else:
                        appe_eval = calculate_appe(
                            self.matrix[valid_m],
                            calc[valid_m]
                        )
                    return appe_eval
                else:
                    return float('inf')

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
                 # Conditional fallback thresholds
                 if self.active_objective_type == 'MSPE':
                     # Thresholds for RMSPE (e.g., 75 means 75% RMSPE)
                     if not best_result_obj.success or best_error_val > 75 and current_method == 'COBYLA': 
                         pass 
                     elif best_error_val > 50 and current_method == 'Nelder-Mead': 
                         pass
                 elif self.active_objective_type == 'WMSE':
                     # Thresholds for MAE + MAPE/10 (e.g., 500)
                     if not best_result_obj.success or best_error_val > 500 and current_method == 'COBYLA':
                         pass
                     elif best_error_val > 250 and current_method == 'Nelder-Mead':
                         pass
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
            final_mse = np.mean(np.square(final_abs_errors))
            final_mae = np.mean(final_abs_errors)
            final_rel_errors = final_abs_errors / np.maximum(1.0, self.matrix[final_valid_mask])
            final_mape = 100 * np.mean(final_rel_errors)
            final_median_ape = np.median(final_abs_errors / np.maximum(self.matrix[final_valid_mask], 1.0)) * 100
            final_max_error = np.max(final_abs_errors)
            final_percentage_errors_for_rmspe = final_abs_errors / np.maximum(self.matrix[final_valid_mask], 1.0)
            final_mspe = np.mean(final_percentage_errors_for_rmspe ** 2)
            final_rmspe = np.sqrt(final_mspe) * 100

        print("\nFinal Error Metrics:")
        print(f"  Mean Absolute Error: ${final_mae:.2f}")
        print(f"  Mean Squared Error: {final_mse:.2f}")
        print(f"  Mean Absolute Percentage Error: {final_mape:.2f}%")
        print(f"  Median Absolute Percentage Error: {final_median_ape:.2f}%")
        print(f"  Root Mean Squared Percentage Error: {final_rmspe:.2f}%")
        print(f"  Max Absolute Error: ${final_max_error:.2f}")

        try:
            print(f"\nSaving results for {objective_display_name} to CSV files...")
            # CSV saving for base rates and discounts can be handled by the caller if needed.
            # pd.DataFrame(final_base_rates, columns=['Base_Rate'], index=[f'Day {i+1}' for i in range(self.n_days)]).to_csv(f'vector{self.objective_filename_suffix}.csv')
            # pd.DataFrame(final_discounts, columns=[f'Cutoff_{c}' for c in self.cutoffs], index=[f'Day {i+1}' for i in range(self.n_days)]).to_csv(f'discounts{self.objective_filename_suffix}.csv')
            
            # Keep comparison data CSV for debugging if desired, or remove if not needed by the caller
            comparison_data = []
            for r_idx in range(self.n_days):
                for c_idx in range(self.n_days):
                    if not np.isnan(self.matrix[r_idx, c_idx]):
                        comparison_data.append({
                            'Day': r_idx + 1,
                            'LOS': c_idx + 1,
                            'OriginalPrice': self.matrix[r_idx, c_idx],
                            'CalculatedPrice': final_calculated_prices[r_idx, c_idx],
                            'AbsoluteError': abs(final_errors[r_idx, c_idx])
                        })
            pd.DataFrame(comparison_data).to_csv(f'price_comparison{self.objective_filename_suffix}.csv', index=False)
            print(f"Price comparison saved to price_comparison{self.objective_filename_suffix}.csv")

            self.plot_results(final_base_rates, final_discounts, final_calculated_prices, filename_suffix=self.objective_filename_suffix)

            # The caller can calculate error_metrics if needed using the returned base_rates and discounts.
            return final_base_rates, final_discounts
        except Exception as e:
            print(f"Error during optimization or result processing in solve method: {e}")
            # Potentially log the error more formally here
            raise # Re-raise the caught exception

    def plot_results(self, base_rates, discounts, calculated, filename_suffix=""):
        """
        Plot the results of the optimization. Appends suffix to filenames.
        """
        print(f"\nGenerating visualizations with suffix: '{filename_suffix}'...")
        try:
            # Original Prices Heatmap (only needs to be saved once, without suffix, or with a generic one if preferred)
            # For simplicity, let's assume it's okay to overwrite or save it once.
            # If it's the first run (e.g., suffix is for MSPE), save it.
            if "_mspe" in filename_suffix or not filename_suffix: # Heuristic to save once
                plt.figure(figsize=(12, 10))
                sns.heatmap(self.matrix, annot=False, fmt=".0f", cmap="viridis", cbar_kws={'label': 'Original Price ($)'})
                plt.title("Original Pricing Matrix")
                plt.xlabel("Length of Stay (Days)")
                plt.ylabel("Start Day")
                plt.savefig(f'original_prices.png') # No suffix for the original, or a generic one
                plt.close()

            # Base Rates Line Plot
            plt.figure(figsize=(12, 6))
            plt.plot(range(1, self.n_days + 1), base_rates, marker='o', linestyle='-')
            plt.title(f"Optimized Base Rates Per Day ({self.active_objective_type})")
            plt.xlabel("Day")
            plt.ylabel("Base Rate ($)")
            plt.grid(True)
            plt.savefig(f'base_rates{filename_suffix}.png')
            plt.close()

            # Discount Curves Line Plot
            plt.figure(figsize=(14, 8))
            for i in range(self.n_cutoffs):
                plt.plot(range(1, self.n_days + 1), discounts[:, i] * 100, marker='.', label=f'LOS Cutoff: {self.cutoffs[i]} days')
            plt.title(f"Optimized Discount Tiers Per Day ({self.active_objective_type})")
            plt.xlabel("Start Day")
            plt.ylabel("Discount (%)")
            plt.legend(title="Discount Tiers", bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f'discount_curves{filename_suffix}.png')
            plt.close()

            # Calculated Prices Heatmap
            plt.figure(figsize=(12, 10))
            sns.heatmap(calculated, annot=False, fmt=".0f", cmap="viridis", cbar_kws={'label': 'Calculated Price ($)'})
            plt.title(f"Calculated Pricing Matrix ({self.active_objective_type})")
            plt.xlabel("Length of Stay (Days)")
            plt.ylabel("Start Day")
            plt.savefig(f'calculated_prices{filename_suffix}.png')
            plt.close()

            # Error Heatmap (Absolute Error)
            error_matrix = calculated - self.matrix
            plt.figure(figsize=(12, 10))
            sns.heatmap(error_matrix, annot=False, fmt=".0f", cmap="coolwarm", center=0, cbar_kws={'label': 'Absolute Error ($)'})
            plt.title(f"Absolute Error Matrix ({self.active_objective_type})")
            plt.xlabel("Length of Stay (Days)")
            plt.ylabel("Start Day")
            plt.savefig(f'error_heatmap{filename_suffix}.png')
            plt.close()
            print(f"Visualizations saved as PNG files (suffix: '{filename_suffix}').")

        except Exception as e:
            print(f"An error occurred during plotting ({self.active_objective_type}): {e}")



def plot_comparison_results(metrics_list, objective_names):
    """
    Plots a comparison of error metrics for different optimization objectives.

    Args:
        metrics_list (list): A list of dictionaries, where each dictionary contains
                             error metrics for one objective.
                             Example: [{'mae': 10, 'mape': 5, ...}, {'mae': 12, 'mape': 6, ...}]
        objective_names (list): A list of names for the objectives, corresponding
                                to the order in metrics_list.
                                Example: ['WMSE', 'MSPE']
    """
    metrics_to_plot = ['mae', 'mape', 'rmspe', 'median_ape', 'max_error']
    metric_labels = {
        'mae': 'Mean Absolute Error ($)',
        'mape': 'Mean Absolute Percentage Error (%)',
        'rmspe': 'Root Mean Squared Percentage Error (%)',
        'median_ape': 'Median Absolute Percentage Error (%)',
        'max_error': 'Max Absolute Error ($)'
    }

    n_objectives = len(metrics_list)
    n_metrics = len(metrics_to_plot)

    # Prepare data for plotting
    plot_data = {}
    for metric_key in metrics_to_plot:
        plot_data[metric_key] = [metrics_dict.get(metric_key, float('nan')) for metrics_dict in metrics_list]

    x = np.arange(n_metrics)  # the label locations for metrics
    total_width = 0.8  # Total width for all bars for a given metric
    bar_width = total_width / n_objectives # the width of individual bars
    
    fig, ax = plt.subplots(figsize=(15, 8)) # Increased figure size for better readability

    for i in range(n_objectives):
        # Calculate offset for each bar group
        offset = (i - (n_objectives - 1) / 2.0) * bar_width
        current_metric_values = [plot_data[metric_key][i] for metric_key in metrics_to_plot]
        rects = ax.bar(x + offset, current_metric_values, bar_width, label=objective_names[i])
        ax.bar_label(rects, padding=3, fmt='%.2f', fontsize=8) # Adjusted font size

    ax.set_ylabel('Error Value', fontsize=12)
    ax.set_title('Comparison of Optimization Objective Error Metrics', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([metric_labels[m] for m in metrics_to_plot], fontsize=10)
    ax.legend(fontsize=10)
    
    plt.xticks(rotation=20, ha="right") # Adjusted rotation
    fig.tight_layout() # Apply tight layout
    
    try:
        plt.savefig('error_metrics_comparison.png')
        print("\nSaved error metrics comparison plot to error_metrics_comparison.png")
    except Exception as e:
        print(f"Error saving comparison plot: {e}")
    finally:
        plt.close(fig)

def main():
    print("Loading pricing matrix from pricing_matrix_30x30.csv...")
    try:
        decomposer = PricingMatrixDecomposer("pricing_matrix_30x30.csv")
        print(f"Matrix shape: {decomposer.matrix.shape}")
        print(f"Number of valid entries: {np.sum(~np.isnan(decomposer.matrix))}")
    except Exception as e:
        print(f"Error loading matrix: {str(e)}")
        import traceback
        traceback.print_exc()
        return

    objective_configs = [
        {
            'type': 'WMSE',
            'suffix': '_wmse',
            'display_name': 'Weighted Mean Squared Error (WMSE)',
            'penalty_coeffs': {
                'non_monotonic': 1e6,
                'regularization': 1e-4,
                'boundary': 1e4
            }
        },
        {
            'type': 'MSPE',
            'suffix': '_mspe',
            'display_name': 'Mean Squared Percentage Error (MSPE)',
            'penalty_coeffs': {
                'non_monotonic': 100.0,
                'regularization': 1e-8,
                'boundary': 1.0
            }
        },
        {
            'type': 'APPE',
            'suffix': '_appe',
            'display_name': 'Asymmetric Powered Percentage Error (APPE k=3, w_neg=2)',
            'appe_params': {'k': 3.0, 'w_neg': 2.0, 'w_pos': 1.0},
            'penalty_coeffs': {
                'non_monotonic': 100.0,
                'regularization': 1e-8,
                'boundary': 100.0
            }
        }
    ]

    all_metrics = []
    objective_names_for_plot = []
    for config in objective_configs:
        print(f"\nStarting optimization for {config['display_name']} (this may take a few minutes)...")
        try:
            result = decomposer.solve(objective_config=config, method='SLSQP', max_iter=1000)
            
            print("\n" + "="*50)
            print(f"OPTIMIZATION RESULTS for {config['display_name']}")
            print("="*50)
            
            print(f"\nOptimization {'succeeded' if result['success'] else 'failed'}")
            if 'message' in result and result['message'] and result['message'] != 'N/A':
                print(f"Message: {result['message']}")
            
            error_metrics = result.get('error_metrics', {})
            print("\nPerformance Metrics:")
            print(f"- Mean Absolute Error (MAE): {error_metrics.get('mae', float('nan')):.2f}")
            print(f"- Mean Squared Error (MSE): {error_metrics.get('mse', float('nan')):.2f}") # Note: This is raw MSE for WMSE, not weighted.
            print(f"- Mean Absolute Percentage Error (MAPE): {error_metrics.get('mape', float('nan')):.2f}%")
            print(f"- Median Absolute Percentage Error (MedAPE): {error_metrics.get('median_ape', float('nan')):.2f}%")
            print(f"- Root Mean Squared Percentage Error (RMSPE): {error_metrics.get('rmspe', float('nan')):.2f}%")
            print(f"- Maximum Absolute Error: {error_metrics.get('max_error', float('nan')):.2f}")
            
            all_metrics.append(error_metrics)
            objective_names_for_plot.append(config['display_name'])
            
            print(f"\nResults for {config['display_name']} have been saved with suffix '{config['suffix']}'.")

        except Exception as e:
            print(f"\nError during optimization for {config['display_name']}: {str(e)}")
            import traceback
            traceback.print_exc()
        print("="*50)

    if len(all_metrics) == len(objective_configs) and len(all_metrics) > 0:
        print("\nGenerating comparison plot for all objectives...")
        plot_comparison_results(all_metrics, objective_names_for_plot)
    else:
        print("\nSkipping comparison plot due to errors in one or more optimization runs.")

    print("\n" + "="*50)
    print("Optimization process completed for all objectives.")
    print("Check individual CSV and PNG files for detailed results for each objective,")
    print("and 'error_metrics_comparison.png' for a summary.")
    print("="*50)

if __name__ == "__main__":
    main()
