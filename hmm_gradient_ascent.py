import numpy as np
#author: Mithi Pandey

# Softmax function with overflow protection
def softmax(x, tau):
    x_max = np.max(x, axis=1, keepdims=True)  # Preventing overflow
    exp_x = np.exp(tau * (x - x_max))  # Subtract max for numerical stability
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Initialize the weights with small values to avoid large gradients
def initialize_params(N, M):
    w = np.random.randn(N, N) * 0.01  # Transition weights
    v = np.random.randn(N, M) * 0.01  # Emission weights
    return w, v

# Compute the expectations for A and B using softmax
def compute_expectations(T, N, observations, w, v, tau):
    Aij = np.zeros((N, N))
    Bij = np.zeros((N, M))
    
    # Compute softmax probabilities for transitions and emissions
    A_softmax = softmax(w, tau)
    B_softmax = softmax(v, tau)
    
    # Approximate gamma_t(i,j) and gamma_t(i)
    gamma_t_ij = np.random.rand(T - 1, N, N)
    gamma_t_i = np.random.rand(T, N)
    
    for t in range(T - 1):
        Aij += gamma_t_ij[t]
    for t in range(T):
        Bij[:, observations[t]] += gamma_t_i[t]
    
    return Aij, Bij

# Update weights using gradient ascent
def update_weights(w, v, Aij, Bij, tau, alpha, C, N, M):
    A_sum = np.sum(Aij, axis=1)
    B_sum = np.sum(Bij, axis=1)
    
    A_softmax = softmax(w, tau)
    B_softmax = softmax(v, tau)
    
    for i in range(N):
        for j in range(N):
            w[i, j] += alpha / C * (Aij[i, j] - A_sum[i] * A_softmax[i, j])
        for j in range(M):
            v[i, j] += alpha / C * (Bij[i, j] - B_sum[i] * B_softmax[i, j])
    
    return w, v

# Train the HMM using gradient ascent
def train_hmm(N, M, T, observations, tau, alpha, epochs):
    w, v = initialize_params(N, M)
    
    for epoch in range(epochs):
        Aij, Bij = compute_expectations(T, N, observations, w, v, tau)
        
        # Making sure that C is not too small for avoiding division by zero and the nan error getting on previous runs
        C = np.max([np.random.rand(), 1e-6])
        
        w, v = update_weights(w, v, Aij, Bij, tau, alpha, C, N, M)
        
        if np.isnan(w).any() or np.isnan(v).any():
            print(f"NaN detected at epoch {epoch}, stopping training")
            break
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: C = {C:.6f}")
    
    return w, v

# Main script
if __name__ == "__main__":
    # Hyperparameters
    N = 2  # Number of hidden states
    M = 27  # Number of emission symbols (26 alpahabets and one for space)
    T = 50000  # Number of observations
    tau = 1.0  # Softmax temperature
    alpha = 0.001  # Learning rate, reduced for stability
    epochs = 100  # Number of training epochs

    # Generate random observations (for the sake of this example)
    observations = np.random.randint(0, M, size=T)
    
    # Train the HMM
    trained_w, trained_v = train_hmm(N, M, T, observations, tau, alpha, epochs)

    # Output the final weights
    print("Final transition weights:", trained_w)
    print("Final emission weights:", trained_v)
