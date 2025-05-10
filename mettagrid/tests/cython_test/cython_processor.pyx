# distutils: language = c++
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False

"""
Python wrapper for C++ ArrayProcessor.

This module provides three different approaches for passing NumPy arrays to C++:
1. Pre-allocated buffer: Reuses the same results array for efficiency
2. New contiguous array: Creates a fresh array each time for flexibility
3. Manual casting: Explicitly controls type conversion for maximum control
"""

import numpy as np
cimport numpy as cnp
from libc.stdint cimport uint8_t, uint32_t

# Import from the .pxd file
from cython_processor cimport ArrayProcessor

# Initialize NumPy
cnp.import_array()

cdef class PyArrayProcessor:
    """
    Python wrapper for C++ ArrayProcessor.
    
    This class provides methods for processing arrays of uint8 values
    using different memory and type handling approaches.
    """
    cdef ArrayProcessor* _cpp_processor
    cdef cnp.ndarray _results_view
    
    def __cinit__(self, uint32_t size):
        """
        Initialize the ArrayProcessor with a specified size.
        
        Args:
            size: The size of arrays this processor will handle
        """
        self._cpp_processor = new ArrayProcessor(size)
        # Create a view of the results buffer
        self._results_view = None
    
    def __dealloc__(self):
        """
        Clean up C++ resources when the Python object is destroyed.
        """
        if self._cpp_processor != NULL:
            del self._cpp_processor
            self._cpp_processor = NULL
        # NumPy array view will be cleaned up automatically
    
    # Method 1: Using a pre-allocated buffer (reused for each call)
    def process_preallocated(self, cnp.ndarray[uint8_t, ndim=1] data):
        """
        Process data using a pre-allocated buffer.
        
        This method reuses the same results array for each call, making it
        efficient for repeated calls on data of the same size.
        
        Pros:
            * Efficient for repeated calls as it reuses the results buffer
            * Minimizes memory allocations
            * Clear type safety with explicit array typing
            
        Cons:
            * The results array is tied to the object's lifetime
            * Less flexible if result size might change
            * Could lead to issues if the view is modified externally
            
        Best for: 
            High-performance code with frequent processing calls on 
            similarly sized data
            
        Args:
            data: Input array of uint8 values. Must be 1D with length matching size.
            
        Returns:
            Array of processed values (doubled input values)
            
        Raises:
            ValueError: If data length doesn't match expected size
        """
        if data.shape[0] != self._cpp_processor.get_size():
            raise ValueError(f"Expected array of size {self._cpp_processor.get_size()}, got {data.shape[0]}")
        
        # Process the array
        self._cpp_processor.process_array(<uint8_t*>data.data, data.shape[0])
        
        # Create a NumPy view of the results if needed
        if self._results_view is None:
            self._results_view = np.zeros(self._cpp_processor.get_size(), dtype=np.uint8)
        
        # Copy results to the view
        cdef uint32_t i
        cdef uint8_t* results = self._cpp_processor.get_results()
        for i in range(self._cpp_processor.get_size()):
            self._results_view[i] = results[i]
        
        return self._results_view
    
    # Method 2: Creating a new contiguous array for each call
    def process_new(self, data):
        """
        Process data by creating a new contiguous array.
        
        This method creates a new array for each call, providing more flexibility
        in handling different input types.
        
        Pros:
            * More flexible input handling (auto-converts types)
            * Clean separation between input and output
            * Each result is independent
            
        Cons:
            * More memory allocations
            * Slightly more overhead for type conversion
            
        Best for:
            General-purpose use where convenience is valued over ultimate performance
            
        Args:
            data: Input data that can be converted to a uint8 NumPy array
            
        Returns:
            New array of processed values (doubled input values)
            
        Raises:
            ValueError: If data length doesn't match expected size
        """
        cdef cnp.ndarray[uint8_t, ndim=1] data_array
        
        # Ensure data is a contiguous array of the right type and shape
        data_array = np.ascontiguousarray(data, dtype=np.uint8)
        
        if data_array.shape[0] != self._cpp_processor.get_size():
            raise ValueError(f"Expected array of size {self._cpp_processor.get_size()}, got {data_array.shape[0]}")
        
        # Process the array
        self._cpp_processor.process_array(<uint8_t*>data_array.data, data_array.shape[0])
        
        # Create a NumPy view of the results
        cdef cnp.ndarray[uint8_t, ndim=1] results_array = np.zeros(self._cpp_processor.get_size(), dtype=np.uint8)
        
        # Copy results to the array
        cdef uint32_t i
        cdef uint8_t* results = self._cpp_processor.get_results()
        for i in range(self._cpp_processor.get_size()):
            results_array[i] = results[i]
        
        return results_array
    
    # Method 3: Taking a buffer and manual casting (closest to your step function)
    def process_manual(self, cnp.ndarray actions):
        """
        Process data with manual copying and casting.
        
        This method provides explicit control over type conversion by manually
        copying and casting each element.
        
        Pros:
            * Maximum control over data conversion
            * Can handle arrays of any type
            * Can perform validation or transformation during copying
            
        Cons:
            * Most verbose approach
            * Extra copying step adds overhead
            * Not needed for many simple cases
            
        Best for:
            Cases where precise control over type conversion is needed
            
        Args:
            actions: Input array of any type
            
        Returns:
            New array of processed values (doubled input values)
        """
        cdef:
            uint32_t size = self._cpp_processor.get_size()
            uint32_t i
            cnp.ndarray[uint8_t, ndim=1] buffer = np.zeros(size, dtype=np.uint8)
        
        # Manual copy with explicit casting
        for i in range(min(size, actions.shape[0])):
            buffer[i] = <uint8_t>actions[i]
        
        # Process the buffer
        self._cpp_processor.process_array(<uint8_t*>buffer.data, size)
        
        # Create results array
        cdef cnp.ndarray[uint8_t, ndim=1] results = np.zeros(size, dtype=np.uint8)
        
        # Copy results
        cdef uint8_t* results_ptr = self._cpp_processor.get_results()
        for i in range(size):
            results[i] = results_ptr[i]
        
        return results