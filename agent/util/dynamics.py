import torch

# AV note: I recommend we work with just singular values, 
# not squared to get "eigenvalues" since it's not a square matrix
def analyze_sv(S: torch.Tensor) -> tuple[float, float]: # the type hint can be changed but we'd have to update how the outputs flow into metta_layer.py and through logging in trainer.py
    ''' Lars, go to town! S is the tensor of singular values. I wrote an example of how to use this function. 
    Right now, it supports outputting two metrics but we can obv add more.
    '''
    eigens = S ** 2
    sorted_eigens = torch.sort(eigens, descending=True).values
    largest_eigen = sorted_eigens[0].item()
    smallest_eigen = sorted_eigens[-1].item()
    return largest_eigen, smallest_eigen