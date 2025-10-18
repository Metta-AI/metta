# MettaGrid Native Socket Server

This document tracks the ongoing work to expose MettaGrid over a language–agnostic socket protocol.
The goal is to allow high-performance gameplay loops without going through Python, making the engine
accessible from other runtimes (Rust, C#, MCP agents, etc.).

## Architecture Overview

- **Transport:** Length-prefixed Protobuf messages over TCP. The default bind address will be `127.0.0.1:5858`
  with configuration overrides in the CLI wrapper.
- **Server runtime:** A single process hosting a `MettaGridGameRegistry`. Each connection can create,
  step, query, and delete multiple game instances by supplying a caller-defined `game_id`.
- **Engine core:** A new `MettaGridEngine` class lives entirely in C++ and manipulates raw buffer views
  (see `env/buffer_views.hpp`). The existing pybind module will eventually delegate to this engine so that
  Python becomes a thin wrapper.
- **Serialization:** Protobuf messages describe `GameConfig`, map layouts, actions, and step results. The
  schema is designed to mirror the current Pydantic models so that we can auto-generate config payloads.

## Request / Response Flow (WIP)

`MettaGridRequest`

- `create_game`: { `game_id`, `config`, `map`, `seed` }
- `step_game`: { `game_id`, `actions` }
- `get_state`: { `game_id` }
- `delete_game`: { `game_id` }

`MettaGridResponse`

- `create_result`: success flag + optional error message
- `step_result`: observations, rewards, terminals, truncations, action success flags
- `state_result`: mirrors `step_result` without advancing the simulation
- `delete_result`: acknowledgement
- `error`: generic failure payload

## Outstanding Tasks

1. Finalize the `.proto` schema and add build rules for C++ code generation.
2. ✅ Implement `MettaGridEngine` that reuses existing gameplay logic without pybind types and route the pybind wrapper through it.
3. Implement the asynchronous TCP server and request dispatcher.
4. Wire reference clients (C++ example + Python/MCP adapters).
5. Add integration tests that hit the socket interface.

This file will be updated as milestones are completed to keep the implementation work aligned with the design.

## Current Implementation Notes

- The Python bindings now delegate to the native `MettaGridEngine`, so the socket server and pybind module share the same gameplay loop.
- Protobuf definitions live in `packages/mettagrid/proto/mettagrid/rpc/v1/mettagrid_service.proto`. Bazel rules generate the corresponding C++ bindings (`mettagrid_rpc_proto_cc`).
- Protobuf definitions live in `packages/mettagrid/proto/mettagrid/rpc/v1/mettagrid_service.proto`. Build with `bazel build //packages/mettagrid/cpp:mettagrid_rpc` (or any target that depends on `mettagrid_rpc_proto_cc`) to regenerate the C++ bindings.
- `mettagrid_rpc` provides conversion helpers, a stateful `GameRegistry`, and a blocking TCP server (`SocketServer`) that speaks the Protobuf protocol over a length-prefixed stream.
- Each socket request carries a `request_id`; responses echo it so clients can correlate asynchronous traffic.
- There is a CLI wrapper (`bazel run //packages/mettagrid/cpp:mettagrid_rpc_server -- --host 0.0.0.0 --port 5858`) that boots the socket server and handles `SIGINT`/`SIGTERM` for clean shutdowns.
- Core lifecycle logic (`GameRegistry`) has a smoke test (`bazel test //packages/mettagrid/cpp:mettagrid_rpc_game_registry_test`) that exercises create/step/delete without going over the network.
