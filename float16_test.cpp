#include <iostream>
#include <iomanip>
#include <bitset>
#include "include/float16.h"

void test_value(float input) {
    try {
        Float16 f16(input);
        float decoded = f16.to_float();
        std::cout << std::scientific << std::setprecision(6)
                  << "Input: " << std::setw(12) << input 
                  << " -> Bits: " << std::bitset<16>(f16.raw_value())
                  << " -> Decoded: " << std::setw(12) << decoded 
                  << " (Error: " << std::fixed << std::abs(input - decoded) << ")\n";
        f16.print_bits();
        std::cout << "---\n";
    } catch (const std::exception& e) {
        std::cout << "Error processing " << input << ": " << e.what() << "\n";
    }
}

void test_multiplication(float a, float b) {
    try {
        Float16 f16_a(a);
        Float16 f16_b(b);
        Float16 result = f16_a * f16_b;
        float expected = a * b;
        float actual = result.to_float();
        
        std::cout << std::scientific << std::setprecision(6)
                  << a << " * " << b << "\n"
                  << "Expected: " << expected << "\n"
                  << "Actual:   " << actual << "\n"
                  << "Error:    " << std::fixed << std::abs(expected - actual) << "\n"
                  << "Bits: ";
        result.print_bits();
        std::cout << "---\n";
    } catch (const std::exception& e) {
        std::cout << "Error in multiplication: " << e.what() << "\n";
    }
}

int main() {
    std::cout << "Basic encoding/decoding tests:\n";
    std::cout << std::string(80, '-') << "\n";
    
    // Test various values
    test_value(1.0f);
    test_value(-1.0f);
    test_value(0.5f);
    test_value(-0.5f);
    test_value(0.333333f);
    test_value(0.123456f);
    test_value(3.14159f);
    test_value(-2.71828f);
    test_value(0.0f);
    
    // Test very small and large numbers
    test_value(1.0e-20f);
    test_value(1.0e20f);
    
    std::cout << "\nMultiplication tests:\n";
    std::cout << std::string(80, '-') << "\n";
    
    test_multiplication(0.5f, 0.5f);
    test_multiplication(-0.5f, 0.5f);
    test_multiplication(0.333333f, 3.0f);
    test_multiplication(1.234f, 5.678f);
    
    return 0;
}