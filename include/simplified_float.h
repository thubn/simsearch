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

public:
  // Default constructor initializes to 0
  SimplifiedFloat() : value(0) {}

  // Constructor from float
  explicit SimplifiedFloat(float num) {
    if (std::abs(num) > 1.0f) {
      throw std::invalid_argument("Number must be between -1 and 1");
    }

    if (num == 0.0f) {
      value = 63;
      return;
    }

    // Extract sign from highest bit
    FloatBits fb{num};
    uint8_t sign = (fb.i >> 31) & 1;

    // Extract biased exponent (8 bits)
    int32_t biased_exp = (fb.i >> 23) & 0xFF;

    // Convert biased exponent to negative power
    // IEEE-754 bias is 127, so an exponent of 126 means 2^-1, 125 means 2^-2,
    // etc.
    uint8_t exponent = static_cast<uint8_t>(127 - biased_exp);

    // Clamp to valid range (0-63)
    if (exponent > 63) {
      exponent = 63;
    }

    // Combine sign and exponent
    value = static_cast<int8_t>((sign << 7) | exponent);
  }

  // Convert back to float
  float to_float() const {
    if (value == 0) {
      return 0.0f;
    }

    // Extract sign and exponent
    bool sign = (value & 0b10000000) != 0;
    uint8_t exponent = value & 0b01111111;

    // Construct float directly using bit manipulation
    FloatBits fb;

    // Set sign bit
    fb.i = static_cast<uint32_t>(sign) << 31;

    // Set biased exponent (127 - exponent)
    // exponent of 1 should give us 2^-1, so biased_exp should be 126 (127-1)
    uint32_t biased_exp = 127 - exponent;
    fb.i |= biased_exp << 23;

    // Set mantissa to 1.0 (implied leading 1 in IEEE-754)
    fb.i |= 0;

    return fb.f;
  }

  // Multiply two SimplifiedFloats
  SimplifiedFloat operator*(const SimplifiedFloat &other) const {
    if (value == 0 || other.value == 0) {
      return SimplifiedFloat();
    }

    // Extract signs and exponents
    uint8_t sign_a = (value & 0b10000000) >> 7;
    uint8_t sign_b = (other.value & 0b10000000) >> 7;
    uint8_t exp_a = value & 0b01111111;
    uint8_t exp_b = other.value & 0b01111111;

    // Calculate new sign and exponent
    uint8_t new_sign = sign_a ^ sign_b; // XOR of signs
    uint8_t exp_sum = static_cast<uint8_t>(exp_a + exp_b);
    uint8_t new_exp = (exp_sum > 63) ? 63 : exp_sum;

    // Create result directly from bits
    SimplifiedFloat result;
    result.value = static_cast<int8_t>((new_sign << 7) | new_exp);
    return result;
  }

  // Get raw int8_t value
  int8_t raw_value() const { return value; }

  // Comparison operators
  bool operator==(const SimplifiedFloat &other) const {
    return value == other.value;
  }

  bool operator!=(const SimplifiedFloat &other) const {
    return !(*this == other);
  }
};