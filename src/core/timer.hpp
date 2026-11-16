#pragma once
// Wall-clock timing with per-phase accumulators (plan D10). Each training step
// adds its compute / communication / waiting time to the matching phase, so a
// run can report where the time went, not only the total.
#include <chrono>
#include <map>
#include <string>

inline double wall_seconds() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

class PhaseTimer {
public:
    void start(const std::string& phase) { started_[phase] = wall_seconds(); }
    void stop(const std::string& phase) { total_[phase] += wall_seconds() - started_[phase]; }
    void add(const std::string& phase, double seconds) { total_[phase] += seconds; }
    double total(const std::string& phase) const {
        auto it = total_.find(phase);
        return it == total_.end() ? 0.0 : it->second;
    }
    const std::map<std::string, double>& totals() const { return total_; }
    void reset() { total_.clear(); }

private:
    std::map<std::string, double> started_;
    std::map<std::string, double> total_;
};
