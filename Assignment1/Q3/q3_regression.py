import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# PART A: Dataset Construction
# ==========================================
# Rule G3: Fix random seeds for reproducibility [cite: 9, 79]
np.random.seed(42)

# A.1. Clean Dataset [cite: 80, 81]
# Formula: y = ax + b + noise
N = 10000 # Sample size >= 10,000 [cite: 78]
a, b = 3.5, 2.0
sigma = 2.0 # Noise standard deviation

x_clean = np.random.uniform(-10, 10, N)
# Add a column of 1s for the bias/intercept term (b)
X_clean = np.c_[np.ones(N), x_clean] 
noise = np.random.normal(0, sigma, N)
y_clean = a * x_clean + b + noise

# A.2. Correlated Feature Dataset [cite: 85, 86]
# Creating 3 features where x2 is highly dependent on x1
x1 = np.random.uniform(-5, 5, N)
x2 = x1 * 2.0 + np.random.normal(0, 0.1, N) # Near-linear function of x1 plus tiny noise
x3 = np.random.uniform(-5, 5, N)

X_corr = np.c_[np.ones(N), x1, x2, x3]
true_weights_corr = np.array([1.5, 2.0, -1.5, 3.0])
y_corr = X_corr @ true_weights_corr + np.random.normal(0, 1.0, N)

# Calculate and print eigenvalues and condition number of X^T X
XtX_corr = X_corr.T @ X_corr
eigenvalues = np.linalg.eigvals(XtX_corr)
condition_number = np.max(eigenvalues) / np.min(eigenvalues)

print("--- Part A.2: Correlated Dataset Stats ---")
print(f"Eigenvalues of X^T X: {eigenvalues}")
print(f"Condition Number: {condition_number:.2f}\n")


# A.3. Outlier Dataset [cite: 87, 88]
# Corrupting 15% of the clean dataset with massive additive deviations
X_outlier = X_clean.copy()
y_outlier = y_clean.copy()

num_outliers = int(0.15 * N) # >= 10% samples [cite: 87]
outlier_indices = np.random.choice(N, num_outliers, replace=False)
# Add a massive deviation to the y-values of the selected indices
y_outlier[outlier_indices] += np.random.normal(50, 20, num_outliers)


# ==========================================
# PART B: From-Scratch Implementations
# ==========================================

# B.1. Closed-form ordinary least squares (OLS) [cite: 92]
def fit_ols_closed_form(X, y):
    """ w = (X^T X)^-1 X^T y """
    # Using np.linalg.solve is computationally more stable than taking the inverse
    return np.linalg.solve(X.T @ X, X.T @ y)

# B.2. Gradient descent for OLS [cite: 94]
def fit_ols_gradient_descent(X, y, lr=0.001, epochs=1000):
    """ w = w - lr * (2/N) * X^T (Xw - y) """
    N_samples, n_features = X.shape
    w = np.zeros(n_features)
    losses = []
    
    for _ in range(epochs):
        predictions = X @ w
        error = predictions - y
        mse_loss = np.mean(error**2)
        losses.append(mse_loss)
        
        gradient = (2/N_samples) * (X.T @ error)
        w -= lr * gradient
        
    return w, losses

# B.3. Ridge regression (Closed-form and Gradient Descent) [cite: 96]
def fit_ridge_closed_form(X, y, lambda_reg):
    """ w = (X^T X + lambda * I)^-1 X^T y """
    n_features = X.shape[1]
    I = np.eye(n_features)
    I[0, 0] = 0 # Do not regularize the intercept
    return np.linalg.solve(X.T @ X + lambda_reg * I, X.T @ y)

def fit_ridge_gradient_descent(X, y, lambda_reg, lr=0.001, epochs=1000):
    """ w = w - lr * [(2/N) * X^T (Xw - y) + 2 * lambda * w] """
    N_samples, n_features = X.shape
    w = np.zeros(n_features)
    
    for _ in range(epochs):
        predictions = X @ w
        error = predictions - y
        
        # Gradient of MSE
        gradient = (2/N_samples) * (X.T @ error)
        
        # Add gradient of L2 penalty (ignoring intercept)
        penalty_gradient = 2 * lambda_reg * w
        penalty_gradient[0] = 0 
        
        w -= lr * (gradient + penalty_gradient)
        
    return w

# B.4. Lasso regression via sub-gradient descent [cite: 98]
def fit_lasso_subgradient(X, y, lambda_reg, lr=0.001, epochs=1000):
    """ w = w - lr * [(2/N) * X^T (Xw - y) + lambda * sign(w)] """
    N_samples, n_features = X.shape
    w = np.zeros(n_features)
    
    for _ in range(epochs):
        predictions = X @ w
        error = predictions - y
        
        gradient = (2/N_samples) * (X.T @ error)
        
        # Sub-gradient for L1 penalty: sign of weights
        penalty_subgradient = lambda_reg * np.sign(w)
        penalty_subgradient[0] = 0 # Do not regularize intercept
        
        w -= lr * (gradient + penalty_subgradient)
        
    return w

# B.5. Weighted least squares (closed-form) [cite: 100]
def fit_weighted_least_squares(X, y, weights):
    """ w = (X^T W X)^-1 X^T W y, where W is a diagonal matrix of weights """
    # For memory efficiency with N=10000, we use broadcasting instead of a giant diagonal matrix
    # X.T @ W is equivalent to multiplying each row of X by its weight
    X_weighted = X * weights[:, np.newaxis] 
    return np.linalg.solve(X.T @ X_weighted, X_weighted.T @ y)

print("Part A and B setup complete. Ready for analysis.")


# ==========================================
# PART C: Gradient Descent Analysis
# ==========================================
print("\n--- Part C: Gradient Descent Analysis ---")

# C.2 Calculate the theoretical maximum learning rate
XtX_clean = X_clean.T @ X_clean
eigenvalues_clean = np.linalg.eigvals(XtX_clean)
lambda_max_clean = np.max(eigenvalues_clean)
# The theoretical bound for convergence is 2 / lambda_max
eta_max = 2 / lambda_max_clean 

print(f"Max Eigenvalue of Clean Data (lambda_max): {lambda_max_clean:.2f}")
print(f"Theoretical Max Learning Rate (2/lambda_max): {eta_max:.6f}")

# C.1 Train with 3 carefully chosen learning rates
# We use the theoretical max to force the three behaviors
lr_converge = eta_max * 0.5   # Safely below the limit
lr_oscillate = eta_max * 0.99 # Right on the edge
lr_diverge = eta_max * 1.05   # Just above the limit

_, losses_conv = fit_ols_gradient_descent(X_clean, y_clean, lr=lr_converge, epochs=50)
_, losses_osc = fit_ols_gradient_descent(X_clean, y_clean, lr=lr_oscillate, epochs=50)
_, losses_div = fit_ols_gradient_descent(X_clean, y_clean, lr=lr_diverge, epochs=50)

# Plotting the three behaviors
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(losses_conv, color='green')
plt.title(f"Converging (lr = {lr_converge:.6f})")
plt.xlabel("Iterations")
plt.ylabel("MSE Loss")

plt.subplot(1, 3, 2)
plt.plot(losses_osc, color='orange')
plt.title(f"Oscillating (lr = {lr_oscillate:.6f})")
plt.xlabel("Iterations")

plt.subplot(1, 3, 3)
plt.plot(losses_div, color='red')
plt.title(f"Diverging (lr = {lr_diverge:.6f})")
plt.xlabel("Iterations")

plt.tight_layout()
plt.savefig("part_c_gradient_descent.png")
print("Saved Part C plot as 'part_c_gradient_descent.png'")


# ==========================================
# PART D: Ill-Conditioning and Ridge Regression
# ==========================================
print("\n--- Part D: Ill-Conditioning and Ridge Regression ---")

lambdas = [0.001, 0.01, 0.1, 1, 10]
weight_magnitudes = []
mses = []

# Fit standard OLS as our baseline
w_ols = fit_ols_closed_form(X_corr, y_corr)
mse_ols = np.mean((X_corr @ w_ols - y_corr)**2)

# Fit Ridge for each lambda
for lambd in lambdas:
    w_ridge = fit_ridge_closed_form(X_corr, y_corr, lambd)
    # np.linalg.norm calculates the overall magnitude (length) of the weight vector
    weight_magnitudes.append(np.linalg.norm(w_ridge)) 
    
    mse_ridge = np.mean((X_corr @ w_ridge - y_corr)**2)
    mses.append(mse_ridge)

# Plotting the effects of Ridge Regression
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(lambdas, weight_magnitudes, marker='o', color='blue')
plt.xscale('log')
plt.axhline(np.linalg.norm(w_ols), color='red', linestyle='--', label='OLS Magnitude')
plt.title("Weight Magnitudes vs. Lambda")
plt.xlabel("Lambda (log scale)")
plt.ylabel("L2 Norm of Weights")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(lambdas, mses, marker='o', color='green')
plt.xscale('log')
plt.axhline(mse_ols, color='red', linestyle='--', label='OLS MSE')
plt.title("Test MSE vs. Lambda")
plt.xlabel("Lambda (log scale)")
plt.ylabel("Mean Squared Error")
plt.legend()

plt.tight_layout()
plt.savefig("part_d_ridge_regression.png")
print("Saved Part D plot as 'part_d_ridge_regression.png'")


# ==========================================
# PART E: Outlier Stress Test
# ==========================================
print("\n--- Part E: Outlier Stress Test ---")

# Train all four models on the corrupted outlier dataset [cite: 124]
w_ols_outlier = fit_ols_closed_form(X_outlier, y_outlier)
w_ols_gd_outlier, _ = fit_ols_gradient_descent(X_outlier, y_outlier, lr=0.001, epochs=1000)
w_ridge_outlier = fit_ridge_closed_form(X_outlier, y_outlier, lambda_reg=10.0)
w_lasso_outlier = fit_lasso_subgradient(X_outlier, y_outlier, lambda_reg=10.0, lr=0.001, epochs=1000)

# Calculate MSE for each 
mse_ols_outlier = np.mean((X_outlier @ w_ols_outlier - y_outlier)**2)
mse_ols_gd_outlier = np.mean((X_outlier @ w_ols_gd_outlier - y_outlier)**2)
mse_ridge_outlier = np.mean((X_outlier @ w_ridge_outlier - y_outlier)**2)
mse_lasso_outlier = np.mean((X_outlier @ w_lasso_outlier - y_outlier)**2)

print(f"OLS (Closed) MSE: {mse_ols_outlier:.2f} | Weights: {w_ols_outlier}")
print(f"OLS (GD) MSE:     {mse_ols_gd_outlier:.2f} | Weights: {w_ols_gd_outlier}")
print(f"Ridge MSE:        {mse_ridge_outlier:.2f} | Weights: {w_ridge_outlier}")
print(f"Lasso MSE:        {mse_lasso_outlier:.2f} | Weights: {w_lasso_outlier}")

# Plotting the regression lines on the scatter plot [cite: 124]
plt.figure(figsize=(10, 6))
plt.scatter(X_outlier[:, 1], y_outlier, color='lightgray', alpha=0.5, label='Data (with outliers)')

# Create a line of x-values for plotting the regression lines smoothly
x_plot = np.linspace(-10, 10, 100)
X_plot = np.c_[np.ones(100), x_plot]

plt.plot(x_plot, X_plot @ w_ols_outlier, label='OLS (Closed Form)', color='blue', linewidth=2)
plt.plot(x_plot, X_plot @ w_ols_gd_outlier, label='OLS (Gradient Descent)', color='cyan', linestyle='--')
plt.plot(x_plot, X_plot @ w_ridge_outlier, label='Ridge (Lambda=10)', color='green', linewidth=2)
plt.plot(x_plot, X_plot @ w_lasso_outlier, label='Lasso (Lambda=10)', color='red', linewidth=2)
# Plot the "true" clean line for reference
plt.plot(x_plot, X_plot @ np.array([b, a]), label='True Clean Line', color='black', linestyle=':', linewidth=3)

plt.title("Part E: Regression Models Under Outlier Stress")
plt.xlabel("Feature (x)")
plt.ylabel("Target (y)")
plt.legend()
plt.tight_layout()
plt.savefig("part_e_outliers.png")
print("Saved Part E plot as 'part_e_outliers.png'")


# ==========================================
# PART F: Iteratively Reweighted Least Squares (IRLS)
# ==========================================
print("\n--- Part F: Iteratively Reweighted Least Squares ---")

N_samples = X_outlier.shape[0]
weights_irls = np.ones(N_samples) # (a) Initialise weights to 1 [cite: 133]
max_iter = 50 # (d) Repeat until convergence or max 50 iterations [cite: 136]

# Store data for plotting specific iterations [cite: 137]
history_w = {}
history_weights = {}
target_iters = [1, 5, 10]

for i in range(1, max_iter + 1):
    # (b) Compute w using weighted closed-form [cite: 134]
    w_irls = fit_weighted_least_squares(X_outlier, y_outlier, weights_irls)
    
    # Save history for required iterations
    if i in target_iters:
        history_w[i] = w_irls.copy()
        history_weights[i] = weights_irls.copy()
        
    # (c) Update weights 
    errors = np.abs(y_outlier - X_outlier @ w_irls)
    new_weights = 1 / (1 + errors)
    
    # Check for convergence (if weights stop changing)
    if np.allclose(weights_irls, new_weights, atol=1e-5):
        print(f"IRLS Converged at iteration {i}")
        break
        
    weights_irls = new_weights

# Save final iteration
final_iter = i
history_w[final_iter] = w_irls.copy()
history_weights[final_iter] = weights_irls.copy()
target_iters.append(final_iter)

# Plotting the IRLS evolution [cite: 137]
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.flatten()

for idx, it in enumerate(target_iters):
    ax = axes[idx]
    current_w = history_w[it]
    current_weights = history_weights[it]
    
    # Scatter plot with color intensity based on the point's weight
    sc = ax.scatter(X_outlier[:, 1], y_outlier, c=current_weights, cmap='viridis', alpha=0.6)
    ax.plot(x_plot, X_plot @ current_w, color='red', linewidth=3, label=f'IRLS Line (Iter {it})')
    ax.plot(x_plot, X_plot @ np.array([b, a]), color='black', linestyle=':', label='True Clean Line')
    
    ax.set_title(f"Iteration {it}")
    ax.legend()
    fig.colorbar(sc, ax=ax, label="Weight $\gamma_i$")

plt.tight_layout()
plt.savefig("part_f_irls.png")
print("Saved Part F plot as 'part_f_irls.png'")