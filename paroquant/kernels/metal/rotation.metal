// Pairwise Givens rotation kernel for Metal (Apple Silicon).
// Template parameters are substituted at JIT compile time by rotation.py.
//
// Grid:  (ceil(batch / ROWS_PER_TILE) * half_group, num_groups, 1)
// Threadgroup: (half_group, 1, 1)

constexpr int ROWS_PER_TILE = {ROWS_PER_TILE};
constexpr int MAX_KROT      = {MAX_KROT};

const int batch_size  = params[0];
const int hidden_size = params[1];
const int krot        = params[2];
const int group_size  = params[3];

const int half_gs     = group_size / 2;
const int half_hidden = hidden_size / 2;

const int tile_idx  = threadgroup_position_in_grid.x;
const int group_idx = threadgroup_position_in_grid.y;
const int tid       = thread_index_in_threadgroup;

if (tid >= half_gs) return;

// ---- Load rotation coefficients into registers ----
float cos_vals[MAX_KROT], sin_vals[MAX_KROT];
int   pair_vals[MAX_KROT];

for (int k = 0; k < krot; k++) {{
    int idx = k * half_hidden + group_idx * half_gs + tid;
    cos_vals[k]  = float(cos_theta[idx]);
    sin_vals[k]  = float(sin_theta[idx]);
    pair_vals[k] = int(packed_pairs[idx]);
}}

// ---- Load activation tile into shared memory (fuse channel scales) ----
threadgroup float tile[{MAX_GROUP_SIZE} * ROWS_PER_TILE];

const int ch_lo = group_idx * group_size + tid;
const int ch_hi = ch_lo + half_gs;
float scale_lo = float(channel_scales[ch_lo]);
float scale_hi = float(channel_scales[ch_hi]);

for (int r = 0; r < ROWS_PER_TILE; r++) {{
    int row = tile_idx * ROWS_PER_TILE + r;
    if (row < batch_size) {{
        tile[tid * ROWS_PER_TILE + r]              = float(x[row * hidden_size + ch_lo]) * scale_lo;
        tile[(tid + half_gs) * ROWS_PER_TILE + r]  = float(x[row * hidden_size + ch_hi]) * scale_hi;
    }}
}}
threadgroup_barrier(mem_flags::mem_threadgroup);

// ---- Apply pairwise Givens rotations in-place ----
for (int k = 0; k < krot; k++) {{
    int i_local = pair_vals[k] & 0xFFFF;
    int j_local = pair_vals[k] >> 16;
    float c = cos_vals[k], s = sin_vals[k];

    for (int m = 0; m < ROWS_PER_TILE; m++) {{
        float a = tile[i_local * ROWS_PER_TILE + m];
        float b = tile[j_local * ROWS_PER_TILE + m];
        tile[i_local * ROWS_PER_TILE + m] = a * c + b * s;
        tile[j_local * ROWS_PER_TILE + m] = b * c - a * s;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
}}

// ---- Write results back ----
for (int r = 0; r < ROWS_PER_TILE; r++) {{
    int row = tile_idx * ROWS_PER_TILE + r;
    if (row < batch_size) {{
        out[row * hidden_size + ch_lo] = tile[tid * ROWS_PER_TILE + r];
        out[row * hidden_size + ch_hi] = tile[(tid + half_gs) * ROWS_PER_TILE + r];
    }}
}}
