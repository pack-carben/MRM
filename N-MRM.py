import numpy as np
import pandas as pd

class NMRM:
    def __init__(self, p, g, beta, n, pi,mean=None,std_dev=None):
        self.p = p
        self.g = g
        self.beta = beta
        self.pi=pi
        self.n = n
        self.mean = np.zeros(g) if mean is None else mean
        self.std_dev = np.ones(g) if std_dev is None else std_dev


        # Check if dimensions match
        assert beta.shape == (g, p+1), "Beta matrix dimensions must be (g ,p+1 ),since we suppose constant term"
        assert len(pi) == g, "Pi vector must have length g"

        # Initialize variables to store generated data
        self.X = None
        self.Y = None

    def generate_data(self, uniform_lower=0, uniform_upper=1.0,random_mode="default",lower_matrix=None,upper_matrix=None,
                      random_state=123):
        np.random.seed(random_state)
        # Calculate each group size
        group_sizes = [int(round(pi_j * self.n)) for pi_j in self.pi]
        group_sizes[-1] = self.n - sum(group_sizes[:-1])

        # Handle lower_matrix and upper_matrix defaults
        if random_mode == "custom":
            if lower_matrix is None:
                lower_matrix = np.full(fill_value=uniform_lower, shape=(self.g, self.p))
            elif lower_matrix.shape != (self.g, self.p):
                raise ValueError("lower_matrix shape does not match (g, p)")

            if upper_matrix is None:
                upper_matrix = np.full(fill_value=uniform_upper, shape=(self.g, self.p))
            elif upper_matrix.shape != (self.g, self.p):
                raise ValueError("upper_matrix shape does not match (g, p)")

        # Determine which random generator function to use
        if random_mode == "default":
            generate_random = lambda j: np.random.uniform(uniform_lower, uniform_upper, size=(group_sizes[j], self.p))
        elif random_mode == "custom":
            generate_random = lambda j: np.random.uniform(lower_matrix[j, :], upper_matrix[j, :],
                                                          size=(group_sizes[j], self.p))
        else:
            raise ValueError("Unsupported random_mode. Choose 'default' or 'custom'.")

        # Generate X matrix with first column as constant term
        self.X = np.ones((self.n, self.p ))
        self.Y = np.zeros(self.n)
        for j in range(self.g):
            start_idx = sum(group_sizes[:j])
            end_idx = start_idx + group_sizes[j]
            beta_group = self.beta[j, 1:]  # Extract betas for this group
            intercept = self.beta[j, 0]    # Extract intercept for this group

            self.X[start_idx:end_idx, :] = generate_random(j)

            # self.X[start_idx:end_idx, 1:] = np.random.uniform(uniform_lower, uniform_upper,
            #                                                   size=(group_sizes[j], self.p))

            self.Y[start_idx:end_idx] = (intercept +
                                         np.dot(self.X[start_idx:end_idx, :], beta_group) +
                                         np.random.normal(loc=self.mean[j], scale=self.std_dev[j], size=group_sizes[j]))

        return self.X, self.Y



# Example usage:

p = 2  # Number of predictors (excluding intercept)
g = 3  # Number of groups
beta = np.array([
    [1.0, 0.5, -0.3],
    [2.0, -1.0, 0.7],
    [0.5, 0.0, 1.2]
])  # Beta matrix with shape (g, p+1)
n = 500  # Total number of observations
pi = np.array([0.3, 0.4, 0.3])  # Proportion vector for each group

# Instantiate NRMM class
nmrm = NMRM(p, g, beta, n, pi)
low=np.array([[0,1],
              [50,100],
              [-100,-50]])
up=np.array([[1,2],
              [51,101],
              [-99,-49]])
# Generate data
X, y = nmrm.generate_data(lower_matrix=low,upper_matrix=up, random_mode = "default ")

# Example output
print("Generated X matrix shape:", X.shape)
print("Generated y vector shape:", y.shape)

# Print first few rows to inspect
print("\nFirst few rows of X:")
print(X.astype(int))
# 合并Y和X为一个二维数组
data = np.column_stack((y, X))
# 创建DataFrame
df = pd.DataFrame(data, columns=['y'] + [f'X{i}' for i in range(1, X.shape[1] + 1)])

# 输出为CSV
df.to_csv('test.csv', index=False)