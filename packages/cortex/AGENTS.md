# Steps to Create and Verify a Triton Kernel for Any Arbitrary Function

1. **Reference implementation in PyTorch**
   Ensure there is a mathematically correct pure-PyTorch implementation of the kernel in `@kernels`.

   * If none exists, extract the relevant computation (cell or block) that requires a Triton kernel and first implement it in `@kernels`.

   Commit the project at this point.

2. **Initial Triton implementation**
   Write a Triton version of the kernel, covering both the forward and backward passes. Refer to other triton to see how they are structured in the project.

   * If a reference Triton implementation has been provided, start from that.

3. **Testing**
   Add a test in `@tests` to check that both the output and gradients of the Triton kernel match the pure-PyTorch reference. Refer to other tests to see how they are structured.

   Commit the project at this point.

4. **Iteration**
   If the test fails, refine the Triton kernel until it passes.

   * **Important:** You may not alter the pure-PyTorch implementation to accommodate the Triton kernel. The PyTorch version is the ground truth.

