import torch
import torch.nn.functional as F
import torchsort
from tensordict import TensorDict
from torch import nn

import metta.agent.lib.nn_layer_library as nn_layer_library


class RobustObsEncoderRev1(nn_layer_library.LayerBase):
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

        self.out_dim = self.eo_dim + self.ea_dim + self.coords_dim  # 8 + 8 + 4 = 20

        # projection layers – query comes from ego-centric center pixels + previous LSTM hidden state
        self.Q = nn.Linear(self.center_pixel_len + self.lstm_h_len, self.score_dim)
        # keys/values are derived from per-feature vectors (without the coordinate sub-vector for projection)
        self.K = nn.Linear(self.eo_dim + self.ea_dim, self.score_dim)
        self.V = nn.Linear(self.eo_dim + self.ea_dim, self.key_dim)

        self.soft_rank_strength = 1.0  # Or from config
        self.soft_rank_temperature = 0.1  # Or from config

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

        query = self.Q(query)  # [N, score_dim]

        device = obs.device

        # container for resulting hidden vectors per batch
        batch_hidden = torch.zeros(N, self.hidden, device=device)

        for b in range(N):
            obs_b = obs[b]  # [C, H, W]
            feats_list = []
            for special_ch, ch_list in self.channel_sets.items():
                # indices where special channel is non-zero
                mask = torch.nonzero(obs_b[special_ch], as_tuple=False)  # [M, 2]
                if mask.numel() == 0:
                    continue

                vals = obs_b[special_ch][mask[:, 0], mask[:, 1]]  # [M]
                # ego-centric coords for these pixels
                coords = self.coords_map[0][mask[:, 0], mask[:, 1]]  # [M, coords_dim]

                # feature embeddings corresponding to this channel set (K == len(ch_list))
                ch_idxs = torch.tensor(ch_list, dtype=torch.long, device=device)
                eo = self.eo_embed(ch_idxs)  # [K, eo_dim]
                ea = self.ea_embed(ch_idxs)  # [K, ea_dim]

                # scale entity-attribute embeddings by pixel value -> [M, K, ea_dim]
                scaled_ea = vals.view(-1, 1, 1) * ea.unsqueeze(0)

                # broadcast entity-object embeddings and coords -> [M, K, ...]
                eo_exp = eo.unsqueeze(0).expand(mask.size(0), -1, -1)
                coords_exp = coords.unsqueeze(1).expand(-1, len(ch_list), -1)

                # concatenate to full feature vector and flatten over pixel+channel dims -> [M*K, out_dim]
                feats = torch.cat([eo_exp, scaled_ea, coords_exp], dim=-1)
                feats_list.append(feats.reshape(-1, self.out_dim))

            if feats_list:
                all_feats = torch.cat(feats_list, dim=0)  # [F, out_dim]
            else:
                all_feats = torch.empty(0, self.out_dim, device=device)

            # compute key & value (ignore coords for projection)
            kv_src = all_feats[:, : self.eo_dim + self.ea_dim]  # [F, 16]
            key = self.K(kv_src)  # [F, score_dim]
            value_core = self.V(kv_src)  # [F, score_dim]
            # append coords back to value
            value = torch.cat([value_core, all_feats[:, self.eo_dim + self.ea_dim :]], dim=1)  # [F, key_dim+4]

            # attention: query vs key – query is single vector, keys are F vectors
            num_features = key.shape[0]
            if num_features == 0:
                attn_scores = torch.empty(0, device=device)
            else:
                attn_scores = torch.matmul(query[b : b + 1], key.T).squeeze(0)  # [num_features]

            # build 8 slices of key_dim length each
            hidden_vec = torch.zeros(self.hidden, device=device)
            T = self.soft_rank_temperature  # Temperature for weighting

            if num_features > 0:
                # Use soft_rank for differentiable ranking (low rank = high score)
                # Add batch dim for torchsort, then remove it
                soft_ranks = torchsort.soft_rank(
                    -attn_scores.unsqueeze(0),  # Negate scores so lower rank is better
                    regularization_strength=self.soft_rank_strength,
                ).squeeze(0)  # [num_features]

                # --- Soft Top-4 Selection ---
                num_top_slots = 4
                # Calculate weights based on proximity to target ranks (0, 1, 2, 3)
                # target_ranks shape [num_top_slots]
                target_ranks = torch.arange(num_top_slots, device=device, dtype=torch.float32)
                # rank_diffs shape [num_features, num_top_slots]
                rank_diffs_sq = (soft_ranks.unsqueeze(-1) - target_ranks) ** 2
                # top_k_weights shape [num_features, num_top_slots]
                top_k_weights = torch.exp(-rank_diffs_sq / T)

                # Normalize weights per target rank (column-wise)
                # Ensure numerical stability for normalization
                norm_top_k_weights = F.normalize(top_k_weights, p=1, dim=0)  # Shape [num_features, num_top_slots]

                # Compute weighted average for each soft top-k slot
                # matmul shape: [num_top_slots, num_features] @ [num_features, key_dim+4] -> [num_top_slots, key_dim+4]
                soft_top_k_values = torch.matmul(norm_top_k_weights.T, value)

                # Fill the first 4 slots of hidden_vec (only key_dim part)
                # Reshape [num_top_slots, key_dim] -> [num_top_slots * key_dim]
                hidden_vec[: num_top_slots * self.key_dim] = soft_top_k_values[:, : self.key_dim].reshape(-1)

                # --- Soft Bucketing for Remaining ---
                if num_features > num_top_slots:
                    num_buckets = 4
                    # Define centers for rank buckets spanning ranks [num_top_slots, F)
                    # Avoid division by zero if num_features is close to num_top_slots
                    rank_range_width = max(1e-6, num_features - num_top_slots)
                    items_per_bucket = rank_range_width / num_buckets
                    bucket_centers = (
                        num_top_slots + (torch.arange(num_buckets, device=device).float() + 0.5) * items_per_bucket
                    )

                    # Calculate weights based on proximity to bucket centers
                    # bucket_rank_diffs shape [num_features, num_buckets]
                    bucket_rank_diffs_sq = (soft_ranks.unsqueeze(-1) - bucket_centers) ** 2
                    # bucket_weights shape [num_features, num_buckets]
                    bucket_weights = torch.exp(-bucket_rank_diffs_sq / T)  # Use same temperature

                    # Normalize weights per bucket (column-wise)
                    norm_bucket_weights = F.normalize(bucket_weights, p=1, dim=0)  # Shape [num_features, num_buckets]

                    # Compute weighted average for each bucket (only key_dim part)
                    # matmul shape: [num_buckets, num_features] @ [num_features, key_dim] -> [num_buckets, key_dim]
                    soft_bucket_values = torch.matmul(norm_bucket_weights.T, value[:, : self.key_dim])

                    # Fill the remaining 4 slots of hidden_vec
                    # Reshape [num_buckets, key_dim] -> [num_buckets * key_dim]
                    hidden_vec[num_top_slots * self.key_dim :] = soft_bucket_values.reshape(-1)
                # Else: Buckets remain zero if num_features <= num_top_slots

            # If num_features == 0, hidden_vec remains zero

            batch_hidden[b] = hidden_vec

        td[self._name] = batch_hidden

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
