#include <iostream>

class StatsTracker {
public:
    void log(const char* msg) { std::cout << msg << std::endl; }
};

class TestClass {
private:
    StatsTracker* stats_tracker;  // Uninitialized!
    int initialized_member;
    
public:
    TestClass() : initialized_member(42) {
        // stats_tracker is NOT initialized here
    }
    
    void use() {
        if (stats_tracker) {  // Using uninitialized pointer
            stats_tracker->log("Using stats");
        }
    }
};

int main() {
    TestClass t;
    t.use();
    return 0;
}
