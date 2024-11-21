#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

class SimplifiedFloat {
private:
  int8_t value; // Internal storage using int8_t

  // Helper union to access float bits
  union FloatBits {
    float f;
    uint32_t i;
  };

  // Constants for bit manipulation
  static constexpr uint8_t SIGN_MASK = 0b10000000;
  static constexpr uint8_t EXP_MASK = 0b01111100;
  static constexpr uint8_t MANTISSA_MASK = 0b00000011;
  static constexpr uint8_t MAX_EXP = 31;  // 5 bits for exponent
  static constexpr uint8_t EXP_SHIFT = 2; // Position of exponent bits
  static constexpr uint8_t MANTISSA_BITS = 2;

public:
  // Default constructor initializes to 0
  SimplifiedFloat() : value(0) {}

  // Constructor from float
  explicit SimplifiedFloat(float num) {
    if (std::abs(num) > 1.0f) {
      throw std::invalid_argument("Number must be between -1 and 1");
    }

    if (num == 0.0f) {
      value = 0;
      return;
    }

    // Extract components from float
    FloatBits fb{num};
    uint8_t sign = (fb.i >> 31) & 1;
    int32_t biased_exp = (fb.i >> 23) & 0xFF;
    uint32_t mantissa = fb.i & 0x7FFFFF;

    // Convert biased exponent to our format
    uint8_t exponent = static_cast<uint8_t>(127 - biased_exp);
    if (exponent > MAX_EXP) {
      exponent = MAX_EXP;
      mantissa = 0;
    }

    // Extract top 2 bits of mantissa (normalize to our 2-bit format)
    uint8_t simplified_mantissa = (mantissa >> 21) & MANTISSA_MASK;

    // Combine components
    value = static_cast<int8_t>(((sign << 7) & SIGN_MASK) |
                                ((exponent << EXP_SHIFT) & EXP_MASK) |
                                (simplified_mantissa & MANTISSA_MASK));
  }

  // Convert back to float
  float to_float() const {
    if (value == 0) {
      return 0.0f;
    }

    // Extract components
    bool sign = (value & SIGN_MASK) != 0;
    uint8_t exponent = (value & EXP_MASK) >> EXP_SHIFT;
    uint8_t mantissa = value & MANTISSA_MASK;

    // Construct float using IEEE-754 format
    FloatBits fb;
    fb.i = 0;

    // Set sign bit
    fb.i |= static_cast<uint32_t>(sign) << 31;

    // Set biased exponent
    uint32_t biased_exp = 127 - exponent;
    fb.i |= biased_exp << 23;

    // Set mantissa (shift to correct position in IEEE-754 format)
    fb.i |= static_cast<uint32_t>(mantissa) << 21;

    return fb.f;
  }

  // Multiply two SimplifiedFloats
  SimplifiedFloat operator*(const SimplifiedFloat &other) const {
    if (value == 0 || other.value == 0) {
      return SimplifiedFloat();
    }

    // Extract components
    uint8_t sign_a = (value & SIGN_MASK) >> 7;
    uint8_t sign_b = (other.value & SIGN_MASK) >> 7;
    uint8_t exp_a = (value & EXP_MASK) >> EXP_SHIFT;
    uint8_t exp_b = (other.value & EXP_MASK) >> EXP_SHIFT;
    uint8_t mantissa_a = value & MANTISSA_MASK;
    uint8_t mantissa_b = other.value & MANTISSA_MASK;

    // Calculate new sign (XOR)
    uint8_t new_sign = sign_a ^ sign_b;

    // Calculate new exponent with clamping
    uint8_t exp_sum = static_cast<uint8_t>(exp_a + exp_b);
    uint8_t new_exp = (exp_sum > MAX_EXP) ? MAX_EXP : exp_sum;

    // Multiply mantissas (normalize back to 2 bits)
    // Shift left by 2 to account for fixed point multiplication
    uint8_t new_mantissa = static_cast<uint8_t>((mantissa_a * mantissa_b) >> 1);
    new_mantissa &= MANTISSA_MASK;

    // Combine components
    SimplifiedFloat result;
    result.value = static_cast<int8_t>(((new_sign << 7) & SIGN_MASK) |
                                       ((new_exp << EXP_SHIFT) & EXP_MASK) |
                                       (new_mantissa & MANTISSA_MASK));
    return result;
  }

  // Get raw int8_t value
  int8_t raw_value() const { return value; }

  // Debug helpers
  void print_bits() const {
    // Extract components
    uint8_t sign = (value & SIGN_MASK) >> 7;
    uint8_t exp = (value & EXP_MASK) >> EXP_SHIFT;
    uint8_t mantissa = value & MANTISSA_MASK;

    printf("Sign: %d, Exp: %02d, Mantissa: %02d (Raw: %02X)\n", sign, exp,
           mantissa, static_cast<uint8_t>(value));
  }

  // Comparison operators
  bool operator==(const SimplifiedFloat &other) const {
    return value == other.value;
  }

  bool operator!=(const SimplifiedFloat &other) const {
    return !(*this == other);
  }
};