#include "util.h"
#include <vector>
#include <stdexcept>
#include <cstdint>
#include <iostream>

std::string readUTF8StringFromFile(const std::string& filename, std::streampos startByte, uint64_t& headerLength) {
    std::ifstream file(filename, std::ios::binary);
    
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file");
    }

    file.read(reinterpret_cast<char*>(&headerLength), sizeof(headerLength));

    file.seekg(startByte);

    std::vector<char> buffer(headerLength);
    size_t bytesRead = file.read(buffer.data(), headerLength).gcount();
    buffer.resize(bytesRead);

    std::string result;
    for (size_t i = 0; i < buffer.size(); ) {
        if ((buffer[i] & 0x80) == 0) {
            result.push_back(buffer[i]);
            i++;
        } else if ((buffer[i] & 0xE0) == 0xC0) {
            if (i + 1 < buffer.size()) {
                result.push_back(buffer[i]);
                result.push_back(buffer[i + 1]);
                i += 2;
            } else break;
        } else if ((buffer[i] & 0xF0) == 0xE0) {
            if (i + 2 < buffer.size()) {
                result.push_back(buffer[i]);
                result.push_back(buffer[i + 1]);
                result.push_back(buffer[i + 2]);
                i += 3;
            } else break;
        } else if ((buffer[i] & 0xF8) == 0xF0) {
            if (i + 3 < buffer.size()) {
                result.push_back(buffer[i]);
                result.push_back(buffer[i + 1]);
                result.push_back(buffer[i + 2]);
                result.push_back(buffer[i + 3]);
                i += 4;
            } else break;
        } else {
            throw std::runtime_error("Invalid UTF-8 sequence encountered");
        }
    }

    return result;
}