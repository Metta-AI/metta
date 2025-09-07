const std = @import("std");

pub const map_agents = 15;

pub const observation_width = 11;

pub const observation_height = 11;

pub const observation_layers = 19;

pub const map_width = 100;

pub const map_height = 50;

pub const SeqInt = opaque {
    extern fn tribal_seq_int_unref(self: *SeqInt) callconv(.C) void;
    pub inline fn deinit(self: *SeqInt) void {
        return tribal_seq_int_unref(self);
    }

    extern fn tribal_new_seq_int() callconv(.C) *SeqInt;
    pub inline fn init() *SeqInt {
        return tribal_new_seq_int();
    }

    extern fn tribal_seq_int_len(self: *SeqInt) callconv(.C) isize;
    pub inline fn len(self: *SeqInt) isize {
        return tribal_seq_int_len(self);
    }

    extern fn tribal_seq_int_get(self: *SeqInt, index: isize) callconv(.C) isize;
    pub inline fn get(self: *SeqInt, index: isize) isize {
        return tribal_seq_int_get(self, index);
    }

    extern fn tribal_seq_int_set(self: *SeqInt, index: isize, value: isize) callconv(.C) void;
    pub inline fn set(self: *SeqInt, index: isize, value: isize) void {
        return tribal_seq_int_set(self, index, value);
    }

    extern fn tribal_seq_int_add(self: *SeqInt, value: isize) callconv(.C) void;
    pub inline fn append(self: *SeqInt, value: isize) void {
        return tribal_seq_int_add(self, value);
    }

    extern fn tribal_seq_int_delete(self: *SeqInt, index: isize) callconv(.C) void;
    pub inline fn remove(self: *SeqInt, index: isize) void {
        return tribal_seq_int_delete(self, index);
    }

    extern fn tribal_seq_int_clear(self: *SeqInt) callconv(.C) void;
    pub inline fn clear(self: *SeqInt) void {
        return tribal_seq_int_clear(self);
    }
};

pub const SeqFloat = opaque {
    extern fn tribal_seq_float_unref(self: *SeqFloat) callconv(.C) void;
    pub inline fn deinit(self: *SeqFloat) void {
        return tribal_seq_float_unref(self);
    }

    extern fn tribal_new_seq_float() callconv(.C) *SeqFloat;
    pub inline fn init() *SeqFloat {
        return tribal_new_seq_float();
    }

    extern fn tribal_seq_float_len(self: *SeqFloat) callconv(.C) isize;
    pub inline fn len(self: *SeqFloat) isize {
        return tribal_seq_float_len(self);
    }

    extern fn tribal_seq_float_get(self: *SeqFloat, index: isize) callconv(.C) f64;
    pub inline fn get(self: *SeqFloat, index: isize) f64 {
        return tribal_seq_float_get(self, index);
    }

    extern fn tribal_seq_float_set(self: *SeqFloat, index: isize, value: f64) callconv(.C) void;
    pub inline fn set(self: *SeqFloat, index: isize, value: f64) void {
        return tribal_seq_float_set(self, index, value);
    }

    extern fn tribal_seq_float_add(self: *SeqFloat, value: f64) callconv(.C) void;
    pub inline fn append(self: *SeqFloat, value: f64) void {
        return tribal_seq_float_add(self, value);
    }

    extern fn tribal_seq_float_delete(self: *SeqFloat, index: isize) callconv(.C) void;
    pub inline fn remove(self: *SeqFloat, index: isize) void {
        return tribal_seq_float_delete(self, index);
    }

    extern fn tribal_seq_float_clear(self: *SeqFloat) callconv(.C) void;
    pub inline fn clear(self: *SeqFloat) void {
        return tribal_seq_float_clear(self);
    }
};

pub const SeqBool = opaque {
    extern fn tribal_seq_bool_unref(self: *SeqBool) callconv(.C) void;
    pub inline fn deinit(self: *SeqBool) void {
        return tribal_seq_bool_unref(self);
    }

    extern fn tribal_new_seq_bool() callconv(.C) *SeqBool;
    pub inline fn init() *SeqBool {
        return tribal_new_seq_bool();
    }

    extern fn tribal_seq_bool_len(self: *SeqBool) callconv(.C) isize;
    pub inline fn len(self: *SeqBool) isize {
        return tribal_seq_bool_len(self);
    }

    extern fn tribal_seq_bool_get(self: *SeqBool, index: isize) callconv(.C) bool;
    pub inline fn get(self: *SeqBool, index: isize) bool {
        return tribal_seq_bool_get(self, index);
    }

    extern fn tribal_seq_bool_set(self: *SeqBool, index: isize, value: bool) callconv(.C) void;
    pub inline fn set(self: *SeqBool, index: isize, value: bool) void {
        return tribal_seq_bool_set(self, index, value);
    }

    extern fn tribal_seq_bool_add(self: *SeqBool, value: bool) callconv(.C) void;
    pub inline fn append(self: *SeqBool, value: bool) void {
        return tribal_seq_bool_add(self, value);
    }

    extern fn tribal_seq_bool_delete(self: *SeqBool, index: isize) callconv(.C) void;
    pub inline fn remove(self: *SeqBool, index: isize) void {
        return tribal_seq_bool_delete(self, index);
    }

    extern fn tribal_seq_bool_clear(self: *SeqBool) callconv(.C) void;
    pub inline fn clear(self: *SeqBool) void {
        return tribal_seq_bool_clear(self);
    }
};

extern fn tribal_check_error() callconv(.C) bool;
pub inline fn checkError() bool {
    return tribal_check_error();
}

extern fn tribal_take_error() callconv(.C) [*:0]const u8;
pub inline fn takeError() [:0]const u8 {
    return std.mem.span(tribal_take_error());
}

pub const TribalGameConfig = extern struct {
    max_steps: isize,
    ore_per_battery: isize,
    batteries_per_heart: isize,
    enable_combat: bool,
    clippy_spawn_rate: f64,
    clippy_damage: isize,
    heart_reward: f64,
    ore_reward: f64,
    battery_reward: f64,
    survival_penalty: f64,
    death_penalty: f64,

    pub fn init(max_steps: isize, ore_per_battery: isize, batteries_per_heart: isize, enable_combat: bool, clippy_spawn_rate: f64, clippy_damage: isize, heart_reward: f64, ore_reward: f64, battery_reward: f64, survival_penalty: f64, death_penalty: f64) TribalGameConfig {
        return TribalGameConfig{
            .max_steps = max_steps,
            .ore_per_battery = ore_per_battery,
            .batteries_per_heart = batteries_per_heart,
            .enable_combat = enable_combat,
            .clippy_spawn_rate = clippy_spawn_rate,
            .clippy_damage = clippy_damage,
            .heart_reward = heart_reward,
            .ore_reward = ore_reward,
            .battery_reward = battery_reward,
            .survival_penalty = survival_penalty,
            .death_penalty = death_penalty,
        };
    }

    extern fn tribal_tribal_game_config_eq(self: TribalGameConfig, other: TribalGameConfig) callconv(.C) bool;
    pub inline fn eql(self: TribalGameConfig, other: TribalGameConfig) bool {
        return tribal_tribal_game_config_eq(self, other);
    }
};

pub const TribalConfig = extern struct {
    game: TribalGameConfig,
    desync_episodes: bool,

    pub fn init(game: TribalGameConfig, desync_episodes: bool) TribalConfig {
        return TribalConfig{
            .game = game,
            .desync_episodes = desync_episodes,
        };
    }

    extern fn tribal_tribal_config_eq(self: TribalConfig, other: TribalConfig) callconv(.C) bool;
    pub inline fn eql(self: TribalConfig, other: TribalConfig) bool {
        return tribal_tribal_config_eq(self, other);
    }
};

pub const TribalEnv = opaque {
    extern fn tribal_tribal_env_unref(self: *TribalEnv) callconv(.C) void;
    pub inline fn deinit(self: *TribalEnv) void {
        return tribal_tribal_env_unref(self);
    }

    extern fn tribal_new_tribal_env(config: TribalConfig) callconv(.C) *TribalEnv;
    /// Create a new tribal environment with full configuration
    pub inline fn init(config: TribalConfig) *TribalEnv {
        return tribal_new_tribal_env(config);
    }

    extern fn tribal_tribal_env_reset_env(self: *TribalEnv) callconv(.C) void;
    /// Reset the environment to initial state
    pub inline fn resetEnv(self: *TribalEnv) void {
        return tribal_tribal_env_reset_env(self);
    }

    extern fn tribal_tribal_env_step(self: *TribalEnv, actions: *SeqInt) callconv(.C) bool;
    /// Step environment with actions
    /// actions: flat sequence of [action_type, argument] pairs
    /// Length should be MapAgents * 2
    pub inline fn step(self: *TribalEnv, actions: *SeqInt) bool {
        return tribal_tribal_env_step(self, actions);
    }

    extern fn tribal_tribal_env_get_observations(self: *TribalEnv) callconv(.C) *SeqInt;
    /// Get current observations as flat sequence
    pub inline fn getObservations(self: *TribalEnv) *SeqInt {
        return tribal_tribal_env_get_observations(self);
    }

    extern fn tribal_tribal_env_get_rewards(self: *TribalEnv) callconv(.C) *SeqFloat;
    /// Get current step rewards for each agent
    pub inline fn getRewards(self: *TribalEnv) *SeqFloat {
        return tribal_tribal_env_get_rewards(self);
    }

    extern fn tribal_tribal_env_get_terminated(self: *TribalEnv) callconv(.C) *SeqBool;
    /// Get terminated status for each agent
    pub inline fn getTerminated(self: *TribalEnv) *SeqBool {
        return tribal_tribal_env_get_terminated(self);
    }

    extern fn tribal_tribal_env_get_truncated(self: *TribalEnv) callconv(.C) *SeqBool;
    /// Get truncated status for each agent
    pub inline fn getTruncated(self: *TribalEnv) *SeqBool {
        return tribal_tribal_env_get_truncated(self);
    }

    extern fn tribal_tribal_env_get_current_step(self: *TribalEnv) callconv(.C) isize;
    /// Get current step number
    pub inline fn getCurrentStep(self: *TribalEnv) isize {
        return tribal_tribal_env_get_current_step(self);
    }

    extern fn tribal_tribal_env_is_episode_done(self: *TribalEnv) callconv(.C) bool;
    /// Check if episode should end
    pub inline fn isEpisodeDone(self: *TribalEnv) bool {
        return tribal_tribal_env_is_episode_done(self);
    }

    extern fn tribal_tribal_env_render_text(self: *TribalEnv) callconv(.C) [*:0]const u8;
    /// Get text rendering of current state
    pub inline fn renderText(self: *TribalEnv) [:0]const u8 {
        return std.mem.span(tribal_tribal_env_render_text(self));
    }
};

extern fn tribal_default_max_steps() callconv(.C) isize;
/// Get default max steps value
pub inline fn defaultMaxSteps() isize {
    return tribal_default_max_steps();
}

extern fn tribal_default_tribal_config() callconv(.C) TribalConfig;
/// Create default tribal configuration
pub inline fn defaultTribalConfig() TribalConfig {
    return tribal_default_tribal_config();
}

