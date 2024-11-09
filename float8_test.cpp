#include "include/simplified_float2.h"
#include <bitset>
#include <iomanip>
#include <iostream>

void test_value(float input) {
  try {
    SimplifiedFloat sf(input);
    float decoded = sf.to_float();
    std::cout << std::scientific << std::setprecision(6)
              << "Input: " << std::setw(12) << input
              << " -> Bits: " << std::bitset<8>(sf.raw_value())
              << " -> Decoded: " << std::setw(12) << decoded << "\n";
    sf.print_bits();
    std::cout << "---\n";
  } catch (const std::exception &e) {
    std::cout << "Error processing " << input << ": " << e.what() << "\n";
  }
}

void test_multiplication(float a, float b) {
  try {
    SimplifiedFloat sf_a(a);
    SimplifiedFloat sf_b(b);
    SimplifiedFloat result = sf_a * sf_b;
    float expected = a * b;
    float actual = result.to_float();

    std::cout << std::scientific << std::setprecision(6) << a << " * " << b
              << "\n"
              << "Expected: " << expected << "\n"
              << "Actual:   " << actual << "\n"
              << "Bits: ";
    result.print_bits();
    std::cout << "---\n";
  } catch (const std::exception &e) {
    std::cout << "Error in multiplication: " << e.what() << "\n";
  }
}

int main() {
  std::cout << "Basic encoding/decoding tests:\n";
  std::cout << std::string(60, '-') << "\n";

  // Test various values
  test_value(0.5f);    // 2^-1
  test_value(-0.5f);   // -2^-1
  test_value(0.25f);   // 2^-2
  test_value(0.3f);    // Tests mantissa
  test_value(0.75f);   // Tests mantissa with different value
  test_value(-0.75f);  // Tests mantissa with sign
  test_value(0.125f);  // 2^-3
  test_value(-0.125f); // -2^-3

  // Test very small numbers (clamping)
  test_value(std::pow(2.0f, -35.0f));

  std::cout << "\nMultiplication tests:\n";
  std::cout << std::string(60, '-') << "\n";

  test_multiplication(0.5f, 0.5f);
  test_multiplication(-0.5f, 0.5f);
  test_multiplication(0.75f, 0.5f);
  test_multiplication(0.3f, 0.3f);

  return 0;
}