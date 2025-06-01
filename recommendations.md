# Proposed Enhanced Optimization Objective: Asymmetric Powered Percentage Error (APPE)

## 1. Introduction: Addressing Refined Optimization Goals

Following our discussion on the Mean Squared Percentage Error (MSPE) performance, it's clear there's an opportunity to further tailor the optimization objective. The key goals you highlighted are:

*   To implement a stronger, "more exponential" penalty for large percentage errors in general (e.g., "cubing the error").
*   To introduce an asymmetric penalty, specifically targeting large negative percentage errors (where the model overprices) more severely, as these are "disastrous," while being more tolerant of "positive" percentage errors (underpricing) if they are rare and not excessively large.

The overall aim is to aggressively reduce large errors, particularly large overpricing instances, even if it means accepting a slightly higher count of smaller, less impactful errors, given the current low average error rate (e.g., 3% MAPE with MSPE).

This document proposes a new objective function, the Asymmetric Powered Percentage Error (APPE), designed to meet these specific requirements.

## 2. Proposed Solution: Asymmetric Powered Percentage Error (APPE)

The core idea is to create a flexible objective function where we can control both the overall aggressiveness of the penalty for large errors and the specific weighting for different types of errors (overpricing vs. underpricing).

Let $P_i$ be the predicted price and $A_i$ be the actual (or target) price for an observation $i$.

The Percentage Error ($PE_i$) is defined as:

$PE_i = \frac{A_i - P_i}{A_i}$

The cost for each observation $i$ within the APPE objective is calculated as follows:

$Cost(PE_i) = \begin{cases} 
w_{neg} \cdot |PE_i|^k & \text{if } PE_i < 0 \text{ (model overprices, i.e., } P_i > A_i\text{)} \\ 
w_{pos} \cdot |PE_i|^k & \text{if } PE_i \geq 0 \text{ (model underprices or predicts perfectly, i.e., } P_i \leq A_i\text{)}
\end{cases}$

The total APPE, which the optimization process will aim to minimize, is the mean of these $Cost(PE_i)$ values across all observations:

$APPE = \frac{1}{N} \sum_{i=1}^{N} Cost(PE_i)$

## 3. Explanation of APPE Components

*   **$k$ (The Power):**
    This exponent determines the "strength" or "aggressiveness" of the penalty for errors of different magnitudes. You suggested "cubing the error," so we can set $k=3$. For an even more pronounced effect on large errors, $k=4$ could also be considered.
    *Impact:* A higher $k$ value makes the objective function much more sensitive to large percentage errors. For instance, with $k=3$, a 30% error is penalized 216 times more than a 5% error, compared to 36 times more with MSPE (where $k=2$). This directly addresses the need for a "more exponential, more strong" penalty.
    This component allows the model to focus on minimizing large deviations, potentially at the cost of a slightly increased number of very small errors (which are minimally penalized when raised to a high power).

*   **$w_{neg}$ and $w_{pos}$ (Asymmetric Weights):**
    These weights allow us to penalize different types of errors differently.
    *   $w_{neg}$: The weight applied to "negative" percentage errors ($PE_i < 0$), which correspond to the model overpricing ($P_i > A_i$). Since these are deemed "disastrous," we will set $w_{neg}$ to be greater than $w_{pos}$.
    *   $w_{pos}$: The weight applied to "positive" percentage errors ($PE_i \geq 0$), corresponding to the model underpricing ($P_i < A_i$) or predicting perfectly ($P_i = A_i$).
    *Example:* We could start with $w_{neg}=3$ and $w_{pos}=1$. This would mean an overpricing percentage error contributes three times more to the loss than an underpricing percentage error of the same magnitude (before the power $k$ is applied). This directly implements the desired asymmetric penalty.

*   **$|PE_i|$ (Absolute Percentage Error):**
    Using the absolute value of the percentage error, $|PE_i|$, ensures that the calculated penalty is always positive and focuses on the magnitude of the error, which is then raised to the power $k$.

## 4. How APPE Addresses Your Requirements

The APPE framework is designed to directly tackle your specified concerns:

*   **Stronger Penalty for Large Errors:** The power $k$ (e.g., 3 or 4) ensures that large PEs are penalized much more heavily than with MSPE.
*   **Asymmetric Penalty for "Negative" Errors (Overpricing):** The $w_{neg} > w_{pos}$ configuration ensures that instances where the model predicts too high a price are penalized more significantly.
*   **Focus on Large Negative Errors:** The combination of a high $k$ and a higher $w_{neg}$ makes large overpricing errors exceptionally costly for the optimizer to ignore. This aligns with your statement, "it would really just be the larger negative errors that we would need to stop," as small errors (even if negative) will have a minimal impact when raised to the power $k$.
*   **Maintained Focus on Percentage Errors:** The foundation of the calculation remains the percentage error, which you found to be a relevant metric.
*   **Leeway for Smaller Errors:** The aggressive penalization of large errors might lead the model to make more very small errors, which is acceptable given the "3% is so low we have some leeway" context.

## 5. Suggested Next Steps & Experimentation

*   **Implementation:** I will modify the existing optimization script to incorporate APPE as an alternative objective function.
*   **Parameter Tuning:** We would then experiment with different configurations for APPE:
    *   Power $k$: Start with $k=3$ (cubed error), potentially explore $k=4$.
    *   Asymmetric Weights: Test ratios for $w_{neg}$ to $w_{pos}$ (e.g., $w_{neg}=2, w_{pos}=1$; or $w_{neg}=3, w_{pos}=1$).
*   **Evaluation:** After running the optimization with the APPE objective, we will re-evaluate all relevant performance metrics (MAE, MAPE, MedAPE, RMSPE, Max Absolute Error) and, importantly, analyze the distribution and nature of the remaining errors (e.g., are large negative PEs significantly reduced?).

## 6. Potential Considerations

*   **Optimization Sensitivity:** Objective functions with higher powers can sometimes lead to steeper or more complex loss landscapes. We'll need to monitor the optimization process for stable convergence.
*   **Error Distribution Shift:** As intended, the model might become more prone to small underpricing errors as it tries to avoid any significant overpricing. This aligns with your stated tolerances but will be a key aspect to observe in the results.

## 7. Conclusion

The Asymmetric Powered Percentage Error (APPE) offers a flexible and powerful way to guide the optimization process more precisely according to your refined business logic for error penalization. It directly translates your preferences for handling large errors, and specifically large overpricing errors, into the mathematical objective of the model.

I look forward to your feedback on this proposal and discussing the best way to proceed with its implementation and testing.
