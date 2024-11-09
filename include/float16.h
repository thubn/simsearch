#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

class Float16 {
private:
  uint16_t value;

  union FloatBits {
    float f;
    uint32_t i;
  };

  static constexpr uint16_t SIGN_MASK = 0b1000000000000000;
  static constexpr uint16_t EXP_MASK = 0b0111111100000000;
  static constexpr uint16_t MANTISSA_MASK = 0b0000000001111111;
  static constexpr uint16_t MAX_EXP = 255;
  static constexpr uint16_t EXP_SHIFT = 7;
  static constexpr uint16_t MANTISSA_BITS = 7;
  static constexpr uint16_t EXP_BIAS = 127;

  static void normalize_mantissa(uint32_t &mantissa, int16_t &exponent) {
    if (mantissa != 0) {
      while (mantissa > 0xFF) {
        mantissa >>= 1;
        exponent++;
      }
      while (mantissa < 0x80) {
        mantissa <<= 1;
        exponent--;
      }
      // Clear the implicit 1
      mantissa &= 0x7F;
    }
  }

public:
  Float16() : value(0) {}

  explicit Float16(float num) {
    FloatBits fb{num};

    if (num == 0.0f || fb.i == 0) {
      value = 0;
      return;
    }

    // Extract components from float32
    uint16_t sign = (fb.i >> 31) & 1;
    int16_t exp = ((fb.i >> 23) & 0xFF); // Keep bias for now
    uint32_t mantissa = (fb.i & 0x7FFFFF);

    // Convert mantissa from 23 to 7 bits
    mantissa = (mantissa + (1 << 15)) >> 16;

    // Handle mantissa overflow
    if (mantissa > 0x7F) {
      mantissa >>= 1;
      exp++;
    }

    // Combine components
    value = static_cast<uint16_t>(((sign << 15) & SIGN_MASK) |
                                  ((exp << EXP_SHIFT) & EXP_MASK) |
                                  (mantissa & MANTISSA_MASK));
  }

  float to_float() const {
    if (value == 0)
      return 0.0f;

    uint32_t sign = (value & SIGN_MASK) >> 15;
    uint32_t exp = (value & EXP_MASK) >> EXP_SHIFT;
    uint32_t mantissa = (value & MANTISSA_MASK);

    // Convert to float32 format
    FloatBits fb;
    fb.i = (sign << 31) | (exp << 23) | (mantissa << 16);

    return fb.f;
  }

  Float16 operator*(const Float16 &other) const {
    if (value == 0 || other.value == 0)
      return Float16();

    // Extract components
    uint16_t sign_a = (value & SIGN_MASK) >> 15;
    uint16_t sign_b = (other.value & SIGN_MASK) >> 15;
    int16_t exp_a = ((value & EXP_MASK) >> EXP_SHIFT);
    int16_t exp_b = ((other.value & EXP_MASK) >> EXP_SHIFT);
    uint32_t mantissa_a = ((value & MANTISSA_MASK) | 0x80); // Add implicit 1
    uint32_t mantissa_b = ((other.value & MANTISSA_MASK) | 0x80);

    // Calculate new sign
    uint16_t new_sign = sign_a ^ sign_b;

    // Multiply mantissas (including hidden bit)
    uint32_t new_mantissa = (mantissa_a * mantissa_b) >>
                            8; // Shift right to account for the extra bits

    // Calculate new exponent (keep bias handling consistent)
    int16_t new_exp = exp_a + exp_b - 127; // Remove one bias

    // Normalize the result
    if (new_mantissa >= 0x80) {
      new_mantissa >>= 1;
      new_exp++;
    }

    // Handle denormals and overflow
    if (new_exp < 0) {
      new_mantissa = 0;
      new_exp = 0;
    } else if (new_exp > 255) {
      new_exp = 255;
      new_mantissa = 0x7F;
    }

    // Clear implicit 1 and combine components
    new_mantissa &= 0x7F;
    Float16 result;
    result.value = static_cast<uint16_t>(((new_sign << 15) & SIGN_MASK) |
                                         ((new_exp << EXP_SHIFT) & EXP_MASK) |
                                         (new_mantissa & MANTISSA_MASK));
    return result;
  }

  uint16_t raw_value() const { return value; }

  void print_bits() const {
    uint16_t sign = (value & SIGN_MASK) >> 15;
    uint16_t exp = (value & EXP_MASK) >> EXP_SHIFT;
    uint16_t mantissa = value & MANTISSA_MASK;
    printf("Sign: %d, Exp: %03d, Mantissa: %03d (Raw: %04X)\n", sign, exp,
           mantissa, value);
  }

  bool operator==(const Float16 &other) const { return value == other.value; }
  bool operator!=(const Float16 &other) const { return !(*this == other); }
};