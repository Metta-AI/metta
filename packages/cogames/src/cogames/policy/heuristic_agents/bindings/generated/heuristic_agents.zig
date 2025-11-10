const std = @import("std");

pub const HeuristicAgent = opaque {
    extern fn heuristic_agents_heuristic_agent_unref(self: *HeuristicAgent) callconv(.C) void;
    pub inline fn deinit(self: *HeuristicAgent) void {
        return heuristic_agents_heuristic_agent_unref(self);
    }

    extern fn heuristic_agents_new_heuristic_agent(agent_id: isize, environment_config: [*:0]const u8) callconv(.C) *HeuristicAgent;
    pub inline fn init(agent_id: isize, environment_config: [:0]const u8) *HeuristicAgent {
        return heuristic_agents_new_heuristic_agent(agent_id, environment_config.ptr);
    }

    extern fn heuristic_agents_heuristic_agent_get_agent_id(self: *HeuristicAgent) callconv(.C) isize;
    pub inline fn getAgentId(self: *HeuristicAgent) isize {
        return heuristic_agents_heuristic_agent_get_agent_id(self);
    }

    extern fn heuristic_agents_heuristic_agent_set_agent_id(self: *HeuristicAgent, value: isize) callconv(.C) void;
    pub inline fn setAgentId(self: *HeuristicAgent, value: isize) void {
        return heuristic_agents_heuristic_agent_set_agent_id(self, value);
    }

    extern fn heuristic_agents_heuristic_agent_reset(self: *HeuristicAgent) callconv(.C) void;
    pub inline fn reset(self: *HeuristicAgent) void {
        return heuristic_agents_heuristic_agent_reset(self);
    }

    extern fn heuristic_agents_heuristic_agent_step(self: *HeuristicAgent, num_agents: isize, num_tokens: isize, size_token: isize, row_observations: pointer, num_actions: isize, raw_actions: pointer) callconv(.C) void;
    pub inline fn step(self: *HeuristicAgent, num_agents: isize, num_tokens: isize, size_token: isize, row_observations: pointer, num_actions: isize, raw_actions: pointer) void {
        return heuristic_agents_heuristic_agent_step(self, num_agents, num_tokens, size_token, row_observations, num_actions, raw_actions);
    }
};

extern fn heuristic_agents_init_chook() callconv(.C) void;
pub inline fn initCHook() void {
    return heuristic_agents_init_chook();
}

