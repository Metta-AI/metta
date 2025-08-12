# AGaLiTe PyTorch Implementation - Refactoring Notes

## Summary of Improvements

### 1. **Cleaner GRU Implementation**
- Kept custom implementation matching JAX formulation exactly
- Maintains orthogonal weight initialization
- Uses simple linear layers instead of trying to adapt PyTorch's GRUCell

### 2. **Extensive Use of `torch.einsum`**
- Replaced verbose tensor operations with einsum notation throughout
- Examples:
  - Outer products: `torch.einsum('chd,chn->chdn', keys, p1)`
  - Oscillatory encoding: `torch.einsum('ci,j->cj', ticks, omegas)`
  - Attention computation: `torch.einsum('crhd,crh->chd', final_values, keys_dot_queries)`

### 3. **Simplified Tensor Operations**
- Used `.view()`, `.flatten()`, and `.unbind()` for cleaner reshaping
- Replaced manual splitting with `.unbind(dim)` for cleaner code
- Used `[..., None]` notation for unsqueezing

### 4. **Improved Batching**
- Simplified BatchedAGaLiTe using list comprehensions
- More efficient memory stacking with dict comprehensions
- Cleaner initialization using nested comprehensions

### 5. **Code Conciseness**
- Reduced redundant code throughout
- Used more Pythonic patterns (comprehensions, unpacking)
- Maintained functionality while improving readability

## Key Design Decisions

1. **Kept JAX Functionality**: As requested, maintained exact JAX behavior rather than strictly following paper
   - Omega values: `linspace(-π, π, r)` instead of `2πk/r`
   - GRU formulation matches JAX exactly

2. **Avoided Over-engineering**: Did not use PyTorch's built-in GRUCell as it would require significant adaptation to match the custom formulation

3. **Performance Considerations**: 
   - Einsum operations are optimized by PyTorch
   - List comprehensions for batching are more efficient than explicit loops
   - Memory operations are vectorized where possible

## Testing
All functionality preserved - tests pass with identical output shapes and behavior.