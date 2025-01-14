import torch
import torch.nn as nn
import torch.nn.functional as F

class HiddenLayerModule(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, i, j, a):
        """
        Initializes:
          - self.fc1 with Xavier uniform (standard practice).
          - self.fc2 such that row i = 0.9*a, row j = -0.8*a, other rows
            random in [-a, a], then we orthogonalize the rows.
        
        Parameters
        ----------
        input_size : int
            Number of input features.
        hidden_size : int
            Number of hidden units.
        output_size : int
            Number of output units.
        i, j : int
            Row indices for the special initialization in fc2.weight.
        a : float
            The 'a' parameter defining the range [-a,a] and the row constants
            (0.9*a and -0.8*a).
        """
        super(HiddenLayerModule, self).__init__()
        
        # Layers
        self.fc1 = nn.Linear(input_size, hidden_size)  # Hidden layer
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size) # Output layer

        # 1) Initialize fc1 with Xavier uniform
        #    (Typically done on the weight matrix; bias often zero or small random)
        nn.init.xavier_uniform_(self.fc1.weight, gain=nn.init.calculate_gain('relu'))
        if self.fc1.bias is not None:
            nn.init.zeros_(self.fc1.bias)

        # 2) Initialize fc2: row i -> 0.9*a, row j -> -0.8*a, others random in [-a,a],
        #    then orthogonalize rows via QR on transpose.
        self._init_fc2_rows(i, j, a)

    def _init_fc2_rows(self, i, j, a):
        """
        Internal helper to set up fc2.weight so that rows are orthonormal after
        starting row i at 0.9*a, row j at -0.8*a, and other rows random in [-a, a].
        """
        # Check that i and j are valid for the row dimension
        out_features, in_features = self.fc2.weight.shape
        if not (0 <= i < out_features) or not (0 <= j < out_features):
            raise ValueError("Row indices i and j must be in [0, out_features-1].")
        if i == j:
            raise ValueError("Row indices i and j must be distinct.")

        # We'll operate in-place on fc2.weight.data
        with torch.no_grad():
            W = self.fc2.weight  # shape: [out_features, in_features]

            # Set row i to 0.9*a
            W[i, :].fill_(0.9 * a)

            # Set row j to -0.8*a
            W[j, :].fill_(-0.8 * a)

            # Fill other rows in [-a,a]
            for row_idx in range(out_features):
                if row_idx not in [i, j]:
                    W[row_idx, :].uniform_(-a, a)

            # Now orthogonalize the rows.
            # We'll do a QR factorization on W^T:
            #    W^T = Q R  =>  W = R^T Q^T
            # That ensures the rows of W become orthonormal.

            # Transpose to shape: [in_features, out_features]
            W_t = W.data.t()  # shape = [in_features, out_features]

            # QR factorization
            # Note: For larger shapes, you might want the "reduced" QR, but
            # torch.qr usually works fine. 
            Q, R = torch.qr(W_t)   # Q: [in_features, in_features], R: [in_features, out_features]

            # Recompute W as R^T @ Q^T => shape [out_features, in_features]
            W_ortho = torch.mm(R.t(), Q.t())

            # Copy back into fc2.weight
            W.copy_(W_ortho)

            print(f"W_ortho row 0: {W_ortho[0, :6]}")
            print(f"W row 0: {W[0, :6]}")
            print(f"W_ortho row 1: {W_ortho[1, :6]}")
            print(f"W row 1: {W[1, :6]}")
            print(f"W_ortho row 2: {W_ortho[2, :6]}")
            print(f"W row 2: {W[2, :6]}")

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# -----------------------------------------------------------------------
# Example usage (if you wanted to try it out):
if __name__ == "__main__":
    model = HiddenLayerModule(
        input_size = 125,
        hidden_size = 512,
        output_size = 20,
        i=0,        # Row 0 => 0.9*a
        j=2,        # Row 2 => -0.8*a
        a=1.0
    )

    # The rows of model.fc2.weight are now orthonormal (up to floating precision).
    # Rows 0 and 2 started as [0.9, 0.9, ..., 0.9] and [-0.8, -0.8, ..., -0.8] but
    # have been adjusted so that the final weight rows are orthonormal.

    # Check row orthonormality:
    W = model.fc2.weight.data  # shape [out_features, in_features]
    row_dot = torch.matmul(W, W.t())  # shape [out_features, out_features]
    print("Row dot products of fc2.weight:\n", row_dot)
    # Expect approx identity on the diagonal, near zero off-diagonal.

    # Create 10 random input vectors
    random_inputs = torch.randn(10, 125)

    # Run each input vector through the model and print the output
    for idx, input_vector in enumerate(random_inputs):
        output = model(input_vector)
        print(f"Output for input vector {idx + 1}:\n{output}\n")


class HiddenLayerModule(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(HiddenLayerModule, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # Hidden layer
        self.relu = nn.ReLU()                         # Activation
        self.fc2 = nn.Linear(hidden_size, output_size)  # Output layer  

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

