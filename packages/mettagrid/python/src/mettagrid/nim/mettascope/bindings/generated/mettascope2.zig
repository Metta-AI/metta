const std = @import("std");

pub const ActionRequest = extern struct {
    agent_id: isize,
    action_id: isize,
    argument: isize,

    pub fn init(agent_id: isize, action_id: isize, argument: isize) ActionRequest {
        return ActionRequest{
            .agent_id = agent_id,
            .action_id = action_id,
            .argument = argument,
        };
    }

    extern fn mettascope2_action_request_eq(self: ActionRequest, other: ActionRequest) callconv(.C) bool;
    pub inline fn eql(self: ActionRequest, other: ActionRequest) bool {
        return mettascope2_action_request_eq(self, other);
    }
};

pub const RenderResponse = opaque {
    extern fn mettascope2_render_response_unref(self: *RenderResponse) callconv(.C) void;
    pub inline fn deinit(self: *RenderResponse) void {
        return mettascope2_render_response_unref(self);
    }

    extern fn mettascope2_render_response_get_should_close(self: *RenderResponse) callconv(.C) bool;
    pub inline fn getShouldClose(self: *RenderResponse) bool {
        return mettascope2_render_response_get_should_close(self);
    }

    extern fn mettascope2_render_response_set_should_close(self: *RenderResponse, value: bool) callconv(.C) void;
    pub inline fn setShouldClose(self: *RenderResponse, value: bool) void {
        return mettascope2_render_response_set_should_close(self, value);
    }

    extern fn mettascope2_render_response_actions_len(self: *RenderResponse) callconv(.C) isize;
    pub inline fn lenActions(self: *RenderResponse) isize {
        return mettascope2_render_response_actions_len(self);
    }

    extern fn mettascope2_render_response_actions_get(self: *RenderResponse, index: isize) callconv(.C) ActionRequest;
    pub inline fn getActions(self: *RenderResponse, index: isize) ActionRequest {
        return mettascope2_render_response_actions_get(self, index);
    }

    extern fn mettascope2_render_response_actions_set(self: *RenderResponse, index: isize, value: ActionRequest) callconv(.C) void;
    pub inline fn setActions(self: *RenderResponse, index: isize, value: ActionRequest) void {
        return mettascope2_render_response_actions_set(self, index, value);
    }

    extern fn mettascope2_render_response_actions_add(self: *RenderResponse, value: ActionRequest) callconv(.C) void;
    pub inline fn appendActions(self: *RenderResponse, value: ActionRequest) void {
        return mettascope2_render_response_actions_add(self, value);
    }

    extern fn mettascope2_render_response_actions_delete(self: *RenderResponse, index: isize) callconv(.C) void;
    pub inline fn removeActions(self: *RenderResponse, index: isize) void {
        return mettascope2_render_response_actions_delete(self, index);
    }

    extern fn mettascope2_render_response_actions_clear(self: *RenderResponse) callconv(.C) void;
    pub inline fn clearActions(self: *RenderResponse) void {
        return mettascope2_render_response_actions_clear(self);
    }
};

extern fn mettascope2_init(data_dir: [*:0]const u8, replay: [*:0]const u8) callconv(.C) *RenderResponse;
pub inline fn init(data_dir: [:0]const u8, replay: [:0]const u8) *RenderResponse {
    return mettascope2_init(data_dir.ptr, replay.ptr);
}

extern fn mettascope2_render(current_step: isize, replay_step: [*:0]const u8) callconv(.C) *RenderResponse;
pub inline fn render(current_step: isize, replay_step: [:0]const u8) *RenderResponse {
    return mettascope2_render(current_step, replay_step.ptr);
}

