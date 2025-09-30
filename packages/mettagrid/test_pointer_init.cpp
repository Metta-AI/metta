#include <iostream>
#include <cstdint>

class TestClass {
public:
    int* ptr1;
    int* ptr2;
    int* ptr3;
    void* ptr4;
    
    TestClass() {
        // Deliberately not initializing any pointers
    }
    
    void print_values() {
        std::cout << "ptr1 = " << static_cast<void*>(ptr1) << " (as int: " << reinterpret_cast<uintptr_t>(ptr1) << ")" << std::endl;
        std::cout << "ptr2 = " << static_cast<void*>(ptr2) << " (as int: " << reinterpret_cast<uintptr_t>(ptr2) << ")" << std::endl;
        std::cout << "ptr3 = " << static_cast<void*>(ptr3) << " (as int: " << reinterpret_cast<uintptr_t>(ptr3) << ")" << std::endl;
        std::cout << "ptr4 = " << ptr4 << " (as int: " << reinterpret_cast<uintptr_t>(ptr4) << ")" << std::endl;
        
        std::cout << "\nChecking if (ptr1): " << (ptr1 ? "true" : "false") << std::endl;
        std::cout << "Checking if (ptr2): " << (ptr2 ? "true" : "false") << std::endl;
        std::cout << "Checking if (ptr3): " << (ptr3 ? "true" : "false") << std::endl;
        std::cout << "Checking if (ptr4): " << (ptr4 ? "true" : "false") << std::endl;
    }
};

int main() {
    std::cout << "=== Stack allocated object ===" << std::endl;
    TestClass stack_obj;
    stack_obj.print_values();
    
    std::cout << "\n=== Heap allocated object ===" << std::endl;
    TestClass* heap_obj = new TestClass();
    heap_obj->print_values();
    delete heap_obj;
    
    return 0;
}
