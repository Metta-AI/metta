#ifndef ARRAY_PROCESSOR_HPP
#define ARRAY_PROCESSOR_HPP

#include <cstdint>
#include <iostream>
#include <stdexcept>

class ArrayProcessor {
public:
  // Constructor
  ArrayProcessor(uint32_t size);

  // Destructor
  ~ArrayProcessor();

  // Process uint8_t array (similar to your step function)
  void process_array(uint8_t* data, uint32_t size);

  // Get results
  uint8_t* get_results() const;

  // Get size
  uint32_t get_size() const;

private:
  uint32_t _size;
  uint8_t* _results;
};

#endif