import torch
import torch.nn as nn
from tensordict import TensorDict

from metta.agent.lib.nn_layer_library import LayerBase


class ObsEncoderRev5_8_01(LayerBase):
    def __init__(self, **cfg):
        super().__init__(**cfg)
        self.channel_sets = {
            0: list(range(0, 14)),
            14: [14, 15],
            16: list(range(16, 27)),
            27: [27] + list(range(19, 27)),
            28: [28] + list(range(17, 27)),
            29: [29] + list(range(17, 27)),
            30: [30] + list(range(17, 27)),
            31: [31] + list(range(17, 27)),
            32: [32] + list(range(17, 27)),
            33: [33] + list(range(17, 27)),
        }

        self.eo_dim = 8
        self.ea_dim = 8
        self.score_dim = 32
        self.key_dim = 16  # dimensionality for key/query/value projections

        self.center_pixel_len = 34
        self.lstm_h_len = 128
        self.lstm_h_layers = 2
        # final hidden vector length (8 slices × key_dim)
        self.hidden = self.key_dim * 8

        self.eo_embed = nn.Embedding(100, self.eo_dim)
        self.ea_embed = nn.Embedding(100, self.ea_dim)

        self.grid_size = (11, 11)
        self.coords_dim = 4
        # precompute ego-centric coordinate map
        H, W = self.grid_size
        coords = torch.zeros(H, W, self.coords_dim)

        for y in range(H):
            for x in range(W):
                coords[y, x] = self.default_warp(y - H // 2, x - W // 2)
        # buffer shape [1, H, W, coords_dim]
        self.register_buffer("coords_map", coords.unsqueeze(0))

        self.out_dim = self.eo_dim + self.ea_dim + self.coords_dim

        # Precompute non-learnable components and indices for feature construction
        self.obj_attr_embed_idx = 99  # Index for special EA embedding

        H, W = self.grid_size

        # Precompute spatial features and radius scale map once for the grid
        # self.coords_map has shape [1, H, W, self.coords_dim]
        # It stores [|dy|, |dx|, log(1+r), 1/(1+r)]
        spatial_features_map_full = self.coords_map.squeeze(0)  # Shape: [H, W, self.coords_dim]
        self.register_buffer("spatial_features_map_full", spatial_features_map_full)

        # The radius scale factor is the 4th element (index 3) of the coords_dim vector
        radius_scale_map = spatial_features_map_full[:, :, 3].unsqueeze(-1)  # Shape: [H, W, 1]
        self.register_buffer("radius_scale_map", radius_scale_map)

        self.channel_set_indices = {}

        for special_channel_key, associated_channels_list in self.channel_sets.items():
            num_assoc_channels = len(associated_channels_list)

            # Store EO index for the special channel key
            # This is a single index, will be used for all associated channels in this group for eo_embed
            sck_eo_idx_tensor = torch.tensor([special_channel_key], dtype=torch.long)
            buffer_name_eo = f"sck_eo_idx_{special_channel_key}"
            self.register_buffer(buffer_name_eo, sck_eo_idx_tensor)

            # Determine EA indices for each channel in the associated list
            ea_indices_list = []
            for channel_val_in_list in associated_channels_list:
                if channel_val_in_list == special_channel_key:
                    # This channel from the list IS the special channel for this group
                    ea_indices_list.append(self.obj_attr_embed_idx)
                else:
                    # This channel from the list is a non-special attribute channel
                    ea_indices_list.append(channel_val_in_list)

            ea_indices_tensor = torch.tensor(ea_indices_list, dtype=torch.long)  # Shape: [num_assoc_channels]
            buffer_name_ea = f"ea_indices_sck{special_channel_key}"
            self.register_buffer(buffer_name_ea, ea_indices_tensor)

            self.channel_set_indices[special_channel_key] = {
                "eo_idx_buffer_name": buffer_name_eo,
                "ea_indices_buffer_name": buffer_name_ea,
                "num_assoc_channels": num_assoc_channels,
                "original_associated_channels_list": associated_channels_list,  # Store for _forward logic mapping to obs
            }

    def _make_net(self):
        # output is a fixed-length hidden vector per batch
        self._out_tensor_shape = [self.hidden]
        return None

    def _forward(self, td: TensorDict):
        obs = td[self._sources[0]["name"]]
        center_pixels = td[self._sources[1]["name"]]
        state_h_prev = td["state_h_prev"]
        N = obs.size(0)

        if state_h_prev is None:
            state_h_prev = torch.zeros(N, self.lstm_h_len, device=obs.device)
        else:
            state_h_prev = state_h_prev[self.lstm_h_layers - 1]

        query = torch.cat([center_pixels, state_h_prev], dim=1)  # [N, center_pixel_len + lstm_h_len]

        expected_dim = self.center_pixel_len + self.lstm_h_len
        assert query.shape == (N, expected_dim), (
            f"Shape mismatch for query: expected ({N}, {expected_dim}), got {query.shape}"
        )
        device = obs.device

        # ----- Run math to get the right feature vectors here ------
        H, W = self.grid_size
        all_processed_features_for_batch = []

        # Retrieve shared non-learnable components (already on the correct device as buffers)
        radius_scale_map_h_w_1 = self.get_buffer("radius_scale_map")
        spatial_features_map_h_w_coords = self.get_buffer("spatial_features_map_full")

        for special_channel_key, group_data in self.channel_set_indices.items():
            eo_idx_buffer_name = group_data["eo_idx_buffer_name"]
            ea_indices_buffer_name = group_data["ea_indices_buffer_name"]
            num_assoc_channels = group_data["num_assoc_channels"]
            original_associated_channels_list = group_data["original_associated_channels_list"]

            # 1. Get Embeddings
            sck_eo_idx = self.get_buffer(eo_idx_buffer_name)  # Shape: [1]
            ea_indices = self.get_buffer(ea_indices_buffer_name)  # Shape: [num_assoc_channels]

            eo_emb = self.eo_embed(sck_eo_idx)  # Shape: [1, eo_dim]
            ea_embs = self.ea_embed(ea_indices)  # Shape: [num_assoc_channels, ea_dim]

            eo_emb_repeated = eo_emb.repeat(num_assoc_channels, 1)  # Shape: [num_assoc_channels, eo_dim]

            semantic_part_unscaled = torch.cat(
                [eo_emb_repeated, ea_embs], dim=-1
            )  # Shape: [num_assoc_channels, eo_dim + ea_dim]
            D_semantic = semantic_part_unscaled.shape[-1]

            # 2. Scale semantic part by radius & concatenate with spatial features
            # Reshape semantic_part for broadcasting with grid: [num_assoc, 1, 1, D_semantic]
            semantic_part_expanded = semantic_part_unscaled.view(num_assoc_channels, 1, 1, D_semantic)
            # radius_scale_map_h_w_1 is [H, W, 1]
            scaled_semantic_grid = (
                semantic_part_expanded * radius_scale_map_h_w_1
            )  # Broadcasts to [num_assoc, H, W, D_semantic]

            # spatial_features_map_h_w_coords is [H, W, coords_dim]
            # Expand for concatenation: [1, H, W, coords_dim] then repeat to [num_assoc, H, W, coords_dim]
            spatial_features_expanded = spatial_features_map_h_w_coords.unsqueeze(0).repeat(num_assoc_channels, 1, 1, 1)

            dynamic_feature_grid = torch.cat([scaled_semantic_grid, spatial_features_expanded], dim=-1)
            # Shape: [num_assoc_channels, H, W, self.out_dim]

            # 3. Process observations for this group
            # obs is [N, C_total, H, W]
            obs_group_slice = obs[:, original_associated_channels_list, :, :]  # Shape: [N, num_assoc_channels, H, W]

            sck_idx_in_slice = -1
            for idx, channel_val in enumerate(original_associated_channels_list):
                if channel_val == special_channel_key:
                    sck_idx_in_slice = idx
                    break

            special_channel_map = obs_group_slice[:, sck_idx_in_slice, :, :].unsqueeze(1)  # Shape: [N, 1, H, W]
            masked_obs_group_slice = obs_group_slice * special_channel_map  # Shape: [N, num_assoc_channels, H, W]

            # 4. Scale dynamic features by masked observation pixel values
            pixel_value_scaler = masked_obs_group_slice.unsqueeze(-1)  # Shape: [N, num_assoc_channels, H, W, 1]

            # dynamic_feature_grid needs to be expanded for batch N: [1, num_assoc, H, W, out_dim]
            output_features_for_group = dynamic_feature_grid.unsqueeze(0) * pixel_value_scaler
            # Shape: [N, num_assoc_channels, H, W, self.out_dim]

            # 5. Reshape and collect
            output_features_for_group_flat = output_features_for_group.view(N, -1, self.out_dim)
            all_processed_features_for_batch.append(output_features_for_group_flat)

        # 6. Concatenate all features from all groups
        if not all_processed_features_for_batch:
            # Handle case with no channel sets, or if all are empty, though unlikely with current setup
            # Output a tensor of shape [N, 0, self.out_dim] or handle as an error/specific case
            # For now, let's assume self.hidden implies a different structure later or this output is intermediate.
            # If an error, obs.device might not be right if obs can be empty. query.device perhaps.
            output = torch.zeros(N, 0, self.out_dim, device=query.device)
        else:
            output = torch.cat(all_processed_features_for_batch, dim=1)
            # Shape: [N, total_locations_across_all_sets, self.out_dim]

        td[self._name] = output

    def default_warp(self, dy: int, dx: int) -> torch.Tensor:
        """
        Default warp function: returns a 4-vector of positive, ego-centric features:
        [|dy|, |dx|, log(1 + r), 1/(1 + r)] where r = sqrt(dy^2 + dx^2).
        """
        # use float tensors for computation
        dy_t = torch.tensor(dy, dtype=torch.float32)
        dx_t = torch.tensor(dx, dtype=torch.float32)
        r = torch.sqrt(dy_t**2 + dx_t**2)
        return torch.stack([torch.abs(dy_t), torch.abs(dx_t), torch.log1p(r), 1.0 / (1.0 + r)])
