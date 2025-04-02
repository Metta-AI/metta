import torch

def analyze_eigen(eigen_values: torch.Tensor) -> dict:
    '''Analyze the eigenvalues of a matrix.'''
    return {
        'min': eigen_values.min().item(),
        'max': eigen_values.max().item(),
        'mean': eigen_values.mean().item(),
    }