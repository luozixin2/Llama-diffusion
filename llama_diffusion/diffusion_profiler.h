// diffusion_profiler.h
#ifndef DIFFUSION_PROFILER_H
#define DIFFUSION_PROFILER_H

#include <chrono>
#include <string>
#include <unordered_map>
#include <vector>
#include <iostream>
#include <iomanip>
#include <algorithm>

namespace diffusion {

class ProfilerTimer {
public:
    ProfilerTimer() : start_(std::chrono::high_resolution_clock::now()) {}
    
    double elapsed_ms() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start_).count();
    }
    
    void reset() {
        start_ = std::chrono::high_resolution_clock::now();
    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
};

struct ProfileStats {
    double total_time_ms = 0.0;
    double min_time_ms = std::numeric_limits<double>::max();
    double max_time_ms = 0.0;
    int call_count = 0;
    std::vector<double> samples;
    
    void add_sample(double time_ms) {
        total_time_ms += time_ms;
        min_time_ms = std::min(min_time_ms, time_ms);
        max_time_ms = std::max(max_time_ms, time_ms);
        call_count++;
        samples.push_back(time_ms);
    }
    
    double avg_time_ms() const {
        return call_count > 0 ? total_time_ms / call_count : 0.0;
    }
    
    double median_time_ms() const {
        if (samples.empty()) return 0.0;
        std::vector<double> sorted = samples;
        std::sort(sorted.begin(), sorted.end());
        size_t mid = sorted.size() / 2;
        if (sorted.size() % 2 == 0) {
            return (sorted[mid - 1] + sorted[mid]) / 2.0;
        }
        return sorted[mid];
    }
    
    double percentile_time_ms(double p) const {
        if (samples.empty()) return 0.0;
        std::vector<double> sorted = samples;
        std::sort(sorted.begin(), sorted.end());
        size_t idx = static_cast<size_t>(p * sorted.size());
        idx = std::min(idx, sorted.size() - 1);
        return sorted[idx];
    }
};

class DiffusionProfiler {
public:
    static DiffusionProfiler& instance() {
        static DiffusionProfiler profiler;
        return profiler;
    }
    
    void start_section(const std::string& name) {
        timers_[name] = ProfilerTimer();
    }
    
    void end_section(const std::string& name) {
        auto it = timers_.find(name);
        if (it != timers_.end()) {
            double elapsed = it->second.elapsed_ms();
            stats_[name].add_sample(elapsed);
            timers_.erase(it);
        }
    }
    
    void record_custom(const std::string& name, double value) {
        custom_metrics_[name].push_back(value);
    }
    
    const std::unordered_map<std::string, ProfileStats>& get_stats() const {
        return stats_;
    }
    
    const std::unordered_map<std::string, std::vector<double>>& get_custom_metrics() const {
        return custom_metrics_;
    }
    
    void reset() {
        stats_.clear();
        timers_.clear();
        custom_metrics_.clear();
    }
    
    void print_report(std::ostream& os = std::cout) const {
        os << "\n" << std::string(100, '=') << "\n";
        os << "DIFFUSION MODEL PERFORMANCE REPORT\n";
        os << std::string(100, '=') << "\n\n";
        
        // Calculate total time
        double grand_total = 0.0;
        for (const auto& pair : stats_) {
            grand_total += pair.second.total_time_ms;
        }
        
        os << std::fixed << std::setprecision(2);
        os << std::left << std::setw(35) << "Section"
           << std::right << std::setw(10) << "Calls"
           << std::setw(12) << "Total(ms)"
           << std::setw(12) << "Avg(ms)"
           << std::setw(12) << "Min(ms)"
           << std::setw(12) << "Max(ms)"
           << std::setw(12) << "P95(ms)" << "\n";
        os << std::string(100, '-') << "\n";
        
        // Sort by total time
        std::vector<std::pair<std::string, ProfileStats>> sorted_stats(stats_.begin(), stats_.end());
        std::sort(sorted_stats.begin(), sorted_stats.end(),
                 [](const auto& a, const auto& b) {
                     return a.second.total_time_ms > b.second.total_time_ms;
                 });
        
        for (const auto& pair : sorted_stats) {
            const auto& name = pair.first;
            const auto& stat = pair.second;
            
            os << std::left << std::setw(35) << name
               << std::right << std::setw(10) << stat.call_count
               << std::setw(12) << stat.total_time_ms
               << std::setw(12) << stat.avg_time_ms()
               << std::setw(12) << stat.min_time_ms
               << std::setw(12) << stat.max_time_ms
               << std::setw(12) << stat.percentile_time_ms(0.95) << "\n";
        }
        
        os << std::string(100, '-') << "\n";
        os << std::left << std::setw(35) << "TOTAL"
           << std::right << std::setw(10) << ""
           << std::setw(12) << grand_total << "\n";
        
        // Custom metrics
        if (!custom_metrics_.empty()) {
            os << "\n" << std::string(100, '=') << "\n";
            os << "CUSTOM METRICS\n";
            os << std::string(100, '=') << "\n\n";
            
            for (const auto& pair : custom_metrics_) {
                const auto& name = pair.first;
                const auto& values = pair.second;
                
                if (!values.empty()) {
                    double sum = 0.0;
                    for (double v : values) sum += v;
                    double avg = sum / values.size();
                    
                    os << std::left << std::setw(40) << name
                       << std::right << std::setw(15) << "Avg: " << std::setw(10) << avg
                       << std::setw(15) << "Count: " << std::setw(10) << values.size() << "\n";
                }
            }
        }
        
        os << "\n" << std::string(100, '=') << "\n\n";
    }
    
    std::unordered_map<std::string, std::unordered_map<std::string, double>> get_summary() const {
        std::unordered_map<std::string, std::unordered_map<std::string, double>> summary;
        
        for (const auto& pair : stats_) {
            const auto& name = pair.first;
            const auto& stat = pair.second;
            
            summary[name]["total_ms"] = stat.total_time_ms;
            summary[name]["avg_ms"] = stat.avg_time_ms();
            summary[name]["min_ms"] = stat.min_time_ms;
            summary[name]["max_ms"] = stat.max_time_ms;
            summary[name]["median_ms"] = stat.median_time_ms();
            summary[name]["p95_ms"] = stat.percentile_time_ms(0.95);
            summary[name]["p99_ms"] = stat.percentile_time_ms(0.99);
            summary[name]["call_count"] = static_cast<double>(stat.call_count);
        }
        
        return summary;
    }

private:
    DiffusionProfiler() = default;
    std::unordered_map<std::string, ProfilerTimer> timers_;
    std::unordered_map<std::string, ProfileStats> stats_;
    std::unordered_map<std::string, std::vector<double>> custom_metrics_;
};

// RAII helper for automatic timing
class ScopedProfiler {
public:
    ScopedProfiler(const std::string& name) : name_(name) {
        DiffusionProfiler::instance().start_section(name_);
    }
    
    ~ScopedProfiler() {
        DiffusionProfiler::instance().end_section(name_);
    }

private:
    std::string name_;
};

#define PROFILE_SECTION(name) diffusion::ScopedProfiler _profiler_##__LINE__(name)

} // namespace diffusion

#endif // DIFFUSION_PROFILER_H
