#include <iostream>

int main()
{
    // define an integer variable named x
    long long x = 0; // this variable is uninitialized because we haven't given it a value
    int y = ++x;


    // print the value of x to the screen
    std::cout << x << '\n'; // who knows what we'll get, because x is uninitialized
    std::cout << sizeof(x) << '\n';

    return 0;
}
