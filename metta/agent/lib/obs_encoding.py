import time

import torch
import torch.nn as nn
from tensordict import TensorDict
from tensordict.tensorclass import NonTensorData

from metta.agent.lib.nn_layer_library import LayerBase

"""
This is experimental code with a number of hardcoded parameters. It expects the LSTM to be 2 layers and 128 wide. The 
number of obs channels should be 34.
"""


class Timer:
    def __init__(self):
        self.sections = {}
        self.start_times = {}

    def start(self, name):
        self.start_times[name] = time.time()

    def stop(self, name):
        if name in self.start_times:
            elapsed_time = time.time() - self.start_times[name]
            self.sections[name] = self.sections.get(name, 0) + elapsed_time
            del self.start_times[name]  # Allow restarting timer for the same section if needed

    def log_all(self, prefix=""):
        # print(f"--- {prefix}Execution Times ---")
        # for name, total_time in self.sections.items():
        #     print(f"{name}: {total_time:.4f}s")
        self.sections.clear()  # Clear after logging to reset for next call if desired


class ObsEmbedder(LayerBase):
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

        self.eo_dim = 4  # get from cfg
        self.ea_dim = 4  # get from cfg
        self.eo_embed = nn.Embedding(100, self.eo_dim)
        self.ea_embed = nn.Embedding(100, self.ea_dim)

        # Initialize embedding weights. I chose larger due to distance scaling.
        nn.init.uniform_(self.eo_embed.weight, -1, 1)
        nn.init.uniform_(self.ea_embed.weight, -1, 1)

        self.grid_size = (11, 11)  # get from cfg
        self.coords_dim = 4  # four so we don't go negative
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
                "original_associated_channels_list": associated_channels_list,  # Store for _forward mapping to obs
            }

    def _make_net(self):
        # output is a fixed-length hidden vector per batch
        self._out_tensor_shape = [self.out_dim]
        self._timer = Timer()  # Initialize timer
        return None

    def _forward(self, td: TensorDict):
        self._timer.start("_forward_total")
        obs = td[self._sources[0]["name"]]
        N = obs.size(0)
        device = obs.device

        H, W = self.grid_size
        all_processed_features_for_batch = []

        self._timer.start("retrieve_shared_buffers")
        # Retrieve shared non-learnable components (already on the correct device as buffers)
        radius_scale_map_h_w_1 = self.get_buffer("radius_scale_map")
        spatial_features_map_h_w_coords = self.get_buffer("spatial_features_map_full")
        self._timer.stop("retrieve_shared_buffers")

        self._timer.start("process_channel_sets_loop")
        for special_channel_key, group_data in self.channel_set_indices.items():
            self._timer.start(f"group_{special_channel_key}_embeddings")
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
            self._timer.stop(f"group_{special_channel_key}_embeddings")

            self._timer.start(f"group_{special_channel_key}_semantic_scaling_concat")
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
            # replace the unsqueeze and repeat op
            spatial_features_expanded = spatial_features_map_h_w_coords.unsqueeze(0).repeat(num_assoc_channels, 1, 1, 1)

            dynamic_feature_grid = torch.cat([scaled_semantic_grid, spatial_features_expanded], dim=-1)
            # Shape: [num_assoc_channels, H, W, self.out_dim]
            self._timer.stop(f"group_{special_channel_key}_semantic_scaling_concat")

            self._timer.start(f"group_{special_channel_key}_obs_processing")
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
            self._timer.stop(f"group_{special_channel_key}_obs_processing")

            self._timer.start(f"group_{special_channel_key}_scale_dynamic_features")
            # 4. Scale dynamic features by masked observation pixel values
            pixel_value_scaler = masked_obs_group_slice.unsqueeze(-1)  # Shape: [N, num_assoc_channels, H, W, 1]

            # dynamic_feature_grid needs to be expanded for batch N: [1, num_assoc, H, W, out_dim]
            output_features_for_group = dynamic_feature_grid.unsqueeze(0) * pixel_value_scaler
            # Shape: [N, num_assoc_channels, H, W, self.out_dim]
            self._timer.stop(f"group_{special_channel_key}_scale_dynamic_features")

            self._timer.start(f"group_{special_channel_key}_reshape_collect")
            # 5. Reshape and collect
            output_features_for_group_flat = output_features_for_group.view(N, -1, self.out_dim)
            all_processed_features_for_batch.append(output_features_for_group_flat)
            self._timer.stop(f"group_{special_channel_key}_reshape_collect")

        self._timer.stop("process_channel_sets_loop")

        self._timer.start("concatenate_all_features")
        # 6. Concatenate all features from all groups
        if not all_processed_features_for_batch:
            # Handle case with no channel sets, or if all are empty, though unlikely with current setup
            # Output a tensor of shape [N, 0, self.out_dim] or handle as an error/specific case
            # For now, let's assume self.hidden implies a different structure later or this output is intermediate.
            # If an error, obs.device might not be right if obs can be empty. query.device perhaps.
            output = torch.zeros(N, 0, self.out_dim, device=device)
        else:
            output = torch.cat(all_processed_features_for_batch, dim=1)
            # Shape: [N, total_locations_across_all_sets, self.out_dim]
        self._timer.stop("concatenate_all_features")

        self._timer.start("generate_attention_mask")
        # output current shape: [N, S, D_feature]
        # S = total_locations_across_all_sets, D_feature = self.out_dim
        N_b, S_orig, D_feat = output.shape  # Capture original N, S, D

        if S_orig > 0:  # Proceed only if there are vectors in the sequence
            # is_zero_vector_mask is True for vectors that ARE all zeros. Shape: [N_b, S_orig]
            is_zero_vector_mask = torch.all(output == 0, dim=2)
            # attention_mask should be True for tokens to ATTEND to (non-zero vectors)
            attention_mask = ~is_zero_vector_mask
            td[self._name + "_attention_mask"] = attention_mask
        else:  # S_orig is 0 (or output was empty initially, N_b might be 0 too)
            # Create an empty mask of shape [N_b, 0]
            # Use output.device as 'device' (obs.device) might not be available if obs was empty
            attention_mask = torch.empty(N_b, S_orig, dtype=torch.bool, device=output.device)
            td[self._name + "_attention_mask"] = attention_mask

        self._timer.stop("generate_attention_mask")

        # Output tensor 'output' remains unchanged in shape [N, S_orig, D_feature]
        # Its content also remains unchanged (includes the original zero vectors)
        # output = output.reshape(-1, self.out_dim) # This was flattening the sequence

        td[self._name] = output
        self._timer.stop("_forward_total")
        self._timer.log_all(prefix=f"{self.__class__.__name__}._forward ")

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


class ObsAttn(LayerBase):
    # if this technique tests well and we can't get sparse obs from MettaGrid then experiment with flex attention as a
    #  speedier alternate
    def __init__(self, **cfg):
        super().__init__(**cfg)
        self.center_pixel_len = 34  # this needs to become another embedded rep. MettaGrid to deliver
        self.lstm_h_len = 128  # get from cfg without making _core_ an input source
        self._query_dim = 32  # get from cfg
        self._lstm_layers = 2  # get from cfg without making _core_ an input source
        self._SQ_proj_hidden = 256  # get from cfg
        self._QV_dim = 32  # get from cfg
        self._V_dim = 128  # get from cfg
        self._out_full_size = 128  # get from cfg

    def _make_net(self):
        self._out_tensor_shape = [self._out_full_size]

        # Assuming self._in_tensor_shapes[0][-1] gives the feature dimension
        # self._feat_dim should be the last dimension of the input tensor shape.
        self._feat_dim = self._in_tensor_shapes[0][-1]  # maybe we should have obs embedder wrap this in the batch dim.

        self._cp_proj = nn.Linear(self.center_pixel_len, self._SQ_proj_hidden)
        self._state_proj = nn.Linear(self.lstm_h_len, self._SQ_proj_hidden)
        # Define the _cat_proj layer
        self._q_vec_proj = nn.Linear(self._SQ_proj_hidden * 2, self._query_dim)
        self._relu = nn.ReLU()

        self._Q = nn.Linear(self._query_dim, self._QV_dim)
        self._K = nn.Linear(self._feat_dim, self._QV_dim)
        self._V = nn.Linear(self._feat_dim, self._V_dim)

        self._timer = Timer()  # Initialize timer
        return None

    def _forward(self, td: TensorDict):
        self._timer.start("_forward_total")
        feat_vectors = td[self._sources[0]["name"]]  # Shape: [B_TT, S, self._feat_dim]
        center_pixels = td[self._sources[1]["name"]]  # Shape: [B_TT, self.center_pixel_len]
        state = td.get("state", None)  # Shape: [B_TT, self.lstm_h_len] (after indexing)
        B_TT = td["_BxTT_"]

        self._timer.start("state_prep")
        # The expand might be redundant if feat_vectors is already [B_TT, S, D]
        # If feat_vectors comes from ObsEncoderRev05_08_01 directly, it is already [B_TT, S, D]
        # For robustness, ensure B_TT matches feat_vectors.shape[0] if not expanding from a shared source.
        # !!! delete this check and handle more robustly !!!
        if feat_vectors.shape[0] == 1 and B_TT > 1:
            feat_vectors = feat_vectors.expand(B_TT, -1, -1)
        elif feat_vectors.shape[0] != B_TT:
            # This case should be handled based on expected input behavior
            # For now, assuming feat_vectors.shape[0] == B_TT if not 1
            pass
        # get_non_tensor()
        if state is None or isinstance(state, NonTensorData):
            # Ensure state_h_prev is correctly initialized for all items in the batch B_TT
            state_h_prev = torch.zeros(B_TT, self.lstm_h_len, device=feat_vectors.device)
        else:
            # Assuming state_h_prev comes from an LSTM with shape [num_layers, B_TT, lstm_h_len]
            state_h_prev = state[self._lstm_layers // 2 - 1]  # Takes the last layer's hidden state
            if state_h_prev.shape[0] != B_TT:
                # Handle cases where state_h_prev might not be per B_TT item (e.g. if from single stream)
                # This might need expansion or specific logic based on how B_TT relates to batching in LSTM
                # For now, assume it aligns or is handled before this layer
                pass
        self._timer.stop("state_prep")

        self._timer.start("query_computation")
        # Prep the input for the Q projection
        cp_proj_out = self._cp_proj(center_pixels)  # Shape: [B_TT, _SQ_proj_hidden]
        state_proj_out = self._state_proj(state_h_prev)  # Shape: [B_TT, _SQ_proj_hidden]

        # Concatenate and project for query
        cat_input = torch.cat([cp_proj_out, state_proj_out], dim=1)  # Shape: [B_TT, 2 * _SQ_proj_hidden]
        cat_input = self._relu(cat_input)
        q_vec_proj_out = self._q_vec_proj(cat_input)  # Shape: [B_TT, _query_dim]
        query_vec = self._relu(q_vec_proj_out)  # query_vec (Query for attention) shape: [B_TT, _query_dim]
        self._timer.stop("query_computation")

        self._timer.start("qkv_projection")
        # Project Q, K, V
        Q_proj = self._Q(query_vec)  # Shape: [B_TT, self._QV_dim]
        K_proj = self._K(feat_vectors)  # Shape: [B_TT, S, self._QV_dim]
        V_proj = self._V(feat_vectors)  # Shape: [B_TT, S, self._V_dim]
        self._timer.stop("qkv_projection")

        self._timer.start("attention_scores")
        # Compute attention scores
        # Q_proj: [B_TT, _QV_dim] -> [B_TT, 1, _QV_dim]
        # K_proj: [B_TT, S, _QV_dim] -> [B_TT, _QV_dim, S] (transposed)
        attn_scores = torch.matmul(Q_proj.unsqueeze(1), K_proj.transpose(-2, -1))  # Shape: [B_TT, 1, S]
        attn_scores = attn_scores.squeeze(1)  # replace the squeeze op # Shape: [B_TT, S]

        # Scale scores
        attn_scores = attn_scores / (self._QV_dim**0.5)
        self._timer.stop("attention_scores")

        self._timer.start("attention_masking")
        # Apply attention mask
        # The mask is expected to be at td[self._sources[0]["name"] + "_attention_mask"]
        # It should have shape [B_TT, S] and be boolean (True for valid tokens)
        mask_key = self._sources[0]["name"] + "_attention_mask"
        if mask_key in td:
            attention_mask = td[mask_key]  # Shape: [B_TT, S]
            # Ensure mask is boolean and on the correct device
            attention_mask = attention_mask.bool().to(attn_scores.device)
            # ~attention_mask is True for tokens to be masked (fill with a large negative number)
            if attn_scores.shape[1] == attention_mask.shape[1]:  # Ensure sequence lengths match
                attn_scores.masked_fill_(~attention_mask, -1e9)
            # else: error or warning, mask shape mismatch
        # else: warning, no attention mask found
        self._timer.stop("attention_masking")

        self._timer.start("attention_softmax_weighted_sum")
        # Softmax to get attention weights
        attn_weights = torch.softmax(attn_scores, dim=-1)  # Shape: [B_TT, S]

        # Compute weighted sum of V
        # attn_weights: [B_TT, S] -> [B_TT, 1, S]
        # V_proj: [B_TT, S, self._V_dim]
        output = torch.matmul(
            attn_weights.unsqueeze(1), V_proj
        )  # replace the unsqueeze op  # Shape: [B_TT, 1, self._V_dim]
        output = output.squeeze(1)  # replace the squeeze op # Shape: [B_TT, self._V_dim]
        self._timer.stop("attention_softmax_weighted_sum")

        td[self._name] = output  # Shape: [B_TT, self._V_dim] (which is 128)
        self._timer.stop("_forward_total")
        self._timer.log_all(prefix=f"{self.__class__.__name__}._forward ")
