#pragma once
// Minimal "--key value" command-line parser shared by every executable.
#include <cstdlib>
#include <map>
#include <stdexcept>
#include <string>

class Args {
public:
    Args(int argc, char** argv) {
        for (int i = 1; i < argc; ++i) {
            std::string key = argv[i];
            if (key.rfind("--", 0) != 0) throw std::runtime_error("expected --key, got " + key);
            key = key.substr(2);
            if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
                values_[key] = argv[++i];
            } else {
                values_[key] = "1";  // flag without a value
            }
        }
    }

    bool has(const std::string& key) const { return values_.count(key) > 0; }

    std::string str(const std::string& key, const std::string& fallback) const {
        auto it = values_.find(key);
        return it == values_.end() ? fallback : it->second;
    }

    long integer(const std::string& key, long fallback) const {
        auto it = values_.find(key);
        return it == values_.end() ? fallback : std::strtol(it->second.c_str(), nullptr, 10);
    }

    double number(const std::string& key, double fallback) const {
        auto it = values_.find(key);
        return it == values_.end() ? fallback : std::strtod(it->second.c_str(), nullptr);
    }

private:
    std::map<std::string, std::string> values_;
};
