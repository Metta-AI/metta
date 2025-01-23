#ifndef EVENT_H
#define EVENT_H

#include <queue>
using namespace std;

#include "grid_object.hpp"

typedef unsigned short EventId;
typedef int EventArg;
struct Event {
    unsigned int timestamp;
    EventId event_id;
    GridObjectId object_id;
    EventArg arg;

    bool operator<(const Event& other) const {
        return timestamp > other.timestamp;
    }
};


#endif // EVENT_H
