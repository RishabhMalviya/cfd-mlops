import torch
import torch.nn as nn
from einops import rearrange


class PhysicsAttentionIrregularMesh(nn.Module):
    """
    This module implements the PhysicsAttention mechanism for irregular meshes 
        from Transolver (https://arxiv.org/pdf/2402.02366). The code is adapted from 
        https://github.com/thuml/Transolver/blob/main/Physics_Attention.py For 
        regular meshes, Transolver uses Conv2d layers for the `in_project*` layers.

        
    Inputs are expected in the shape (B, N, C) where B is batch size, N 
        is the number of (irregular) nodes, and C is the node feature dimension.
    """

    def __init__(self, in_out_dim, num_attn_heads=8, attn_head_dim=64, dropout=0., num_slices=64):
        super().__init__()
        self.attn_head_dim = attn_head_dim
        self.num_attn_heads = num_attn_heads
        self.inner_dim = self.attn_head_dim * self.num_attn_heads

        self.scale = self.attn_head_dim ** -0.5

        self.dropout = nn.Dropout(dropout)

        self.softmax = nn.Softmax(dim=-1)
        self.softmax_temperature_scaling = nn.Parameter(torch.ones([1, self.num_attn_heads, 1, 1]) * 0.5)

        self.in_project_x = nn.Linear(in_out_dim, self.inner_dim)
        self.in_project_fx = nn.Linear(in_out_dim, self.inner_dim)
        self.in_project_slice = nn.Linear(self.attn_head_dim, num_slices)
        for l in [self.in_project_slice]:
            torch.nn.init.orthogonal_(l.weight)  # use a principled initialization
        self.to_q = nn.Linear(self.attn_head_dim, self.attn_head_dim, bias=False)
        self.to_k = nn.Linear(self.attn_head_dim, self.attn_head_dim, bias=False)
        self.to_v = nn.Linear(self.attn_head_dim, self.attn_head_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(self.inner_dim, in_out_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # B N C
        B, N, C = x.shape

        ## 1 SLICE
        fx_mid = self.in_project_fx(x).reshape(B, N, self.num_attn_heads, self.attn_head_dim) \
            .permute(0, 2, 1, 3).contiguous()  # B H N C
        x_mid = self.in_project_x(x).reshape(B, N, self.num_attn_heads, self.attn_head_dim) \
            .permute(0, 2, 1, 3).contiguous()  # B H N C
        slice_weights = self.softmax(self.in_project_slice(x_mid) / self.softmax_temperature_scaling)  # B H N G
        slice_norm = slice_weights.sum(2)  # B H G
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.attn_head_dim))

        ## 2 ATTENTION
        q_slice_token = self.to_q(slice_token)
        k_slice_token = self.to_k(slice_token)
        v_slice_token = self.to_v(slice_token)
        dots = torch.matmul(q_slice_token, k_slice_token.transpose(-1, -2)) * self.scale
        attn = self.softmax(dots)
        attn = self.dropout(attn)
        out_slice_token = torch.matmul(attn, v_slice_token)  # B H G D

        # 3 DESLICE
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
        out_x = rearrange(out_x, 'b h n d -> b n (h d)')
        return self.to_out(out_x)
    

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=1):
        super(MLP, self).__init__()

        self.n_layers = n_layers

        self.linear_pre = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU())
        self.linears = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU())
            for _ in range(n_layers)
        ])
        self.linear_post = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.linear_pre(x)
        for i in range(self.n_layers):
            x = self.linears[i](x)
        x = self.linear_post(x)

        return x


class TransolverBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(
            self,
            hidden_dim: int,
            num_attn_heads: int,
            dropout: float,
            mlp_ratio=4,
            is_output_block=False,
            out_dim=1,
            num_slices=32
    ):
        super().__init__()
        assert hidden_dim % num_attn_heads == 0, "hidden_dim must be divisible by num_attn_heads"

        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.PhysicsAttention = PhysicsAttentionIrregularMesh(
            in_out_dim=hidden_dim,
            num_attn_heads=num_attn_heads,
            attn_head_dim=hidden_dim // num_attn_heads,
            dropout=dropout,
            num_slices=num_slices
        )

        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(
            input_dim=hidden_dim, 
            hidden_dim=hidden_dim * mlp_ratio, 
            output_dim=hidden_dim, 
            n_layers=0
        )

        self.return_out_dim = is_output_block
        if self.return_out_dim:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.PhysicsAttention(self.ln_1(x)) + x
        x = self.mlp(self.ln_2(x)) + x
        if self.return_out_dim:
            x = self.mlp2(self.ln_3(x))

        return x


class Model(nn.Module):
    def __init__(self,
        n_layers=5,
        hidden_dim=256,
        dropout=0,
        n_head=8, 
        mlp_ratio=1,
        in_dim=6,  # 3 for position, 3 for normals
        out_dim=4, # 3 for wall shear stress, 1 for pressure
        num_slices=32
    ):
        super(Model, self).__init__()

        self.latent_rep_layer = MLP(
            input_dim=in_dim,
            hidden_dim=hidden_dim * 2,
            output_dim=hidden_dim,
            n_layers=0
        )
        self.latent_rep_layer_bias = nn.Parameter((1 / (hidden_dim)) * torch.rand(hidden_dim, dtype=torch.float))

        self.blocks = nn.ModuleList([
            TransolverBlock(hidden_dim=hidden_dim,
                            num_attn_heads=n_head,
                            dropout=dropout,
                            mlp_ratio=mlp_ratio,
                            out_dim=out_dim,
                            num_slices=num_slices,
                            is_output_block=(layer_idx == n_layers - 1)
            )
            for layer_idx in range(n_layers)
        ])

        self.initialize_weights()
        
    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        std = 0.02
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=std, a=-2.0*std, b=2.0*std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):  # the input `x` is the `x` from the `pytorch_geometric.data.Data` object, which is of shape (N, 6) for the 3D meshes in DrivAer
        # x = x[None, :, :]  # Adding the batch dimension, so x is of shape (1, N, 6)

        x = self.latent_rep_layer(x) + self.latent_rep_layer_bias[None, None, :]

        for block in self.blocks:
            x = block(x)

        return x[0]
    

if __name__ == "__main__":
    import os
    from drivaer_dataset import DrivAerDataset

    # --- Test Model Instantiation ---
    in_dim = 6
    out_dim = 4

    model = Model(
        n_layers=5,
        hidden_dim=256,
        dropout=0,
        n_head=8, 
        mlp_ratio=1,
        in_dim=in_dim,
        out_dim=out_dim,
        num_slices=32
    )
    print(model)


    # --- Test Forward Pass ---
    print('Testing forward pass with random input...')

    N = 10_000
    x = torch.rand(N, 6)
    print(f"Input shape : {x.shape} (`in_dim` for model was {in_dim})")

    with torch.no_grad():
        out = model(x)
    print(f"Output shape: {out.shape} (`out_dim` for model was {out_dim})")


    # --- Test Forward Pass With DrivaerDataset ---
    # TODO: Try with decimated input meshes (e.g., 10k nodes) to see if the model can handle it.
    # The current DrivAer meshes have ~100k nodes, which is too large for the model to handle on a single GPU.   
    print('Testing forward pass with DrivAerDataset input on GPU...')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    data_dir  = "./data/drivaer_data"
    # avail_run_ids   = [i for i in list(range(1,51)) if os.path.exists(os.path.join(data_dir, f"run_{i}", f"boundary_{i}.vtp"))]
    avail_run_ids = [1,2]
    ds = DrivAerDataset(data_dir=data_dir, run_ids=avail_run_ids)
    x = ds[0].x
    x = x.to(device)
    print(f"Input shape : {x.shape} (`in_dim` for model was {in_dim})")
    
    with torch.no_grad():
        out = model(x)
    print(f"Output shape: {out.shape} (`out_dim` for model was {out_dim})")
