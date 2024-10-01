#ifndef UTF8_STRING_READER_H
#define UTF8_STRING_READER_H

#include <string>
#include <fstream>
#include <cstdint>

/**
 * @brief Reads a UTF-8 encoded string from a binary file.
 * 
 * @param filename The path to the file to read from.
 * @param startByte The byte position in the file to start reading from.
 * @param maxBytes The maximum number of bytes to read.
 * @return std::string The UTF-8 encoded string read from the file.
 * @throws std::runtime_error if the file cannot be opened or if an invalid UTF-8 sequence is encountered.
 */
std::string readUTF8StringFromFile(const std::string& filename, std::streampos startByte, uint64_t& headerLength);

#endif // UTF8_STRING_READER_H