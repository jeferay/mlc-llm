import dataclasses
from typing import Tuple, Optional
import torch
import tvm
from tvm import te, tir
from tvm.script import relax as R
import tvm.relax as relax
import tvm.relax.frontend.nn as nn
from tvm.relax.frontend.nn.modules import (
    Conv2D,
    Parameter,
    Embedding,
    ModuleList,
    LayerNorm,
    Linear
)
from tvm.relax.frontend.nn import(
    Tensor,
    Module
)
from tvm.relax.frontend.nn.op import(
    reshape,
    permute_dims,
    repeat,
    broadcast_to,
    concat,
    softmax,
    matmul,
    _wrap_nested
)
import tvm.relax.frontend.nn.spec as spec

@dataclasses.dataclass
class LlavaConfig:
    image_size: int = 224
    num_channels: int = 3
    hidden_size: int = 1024
    projection_dim: int = 768
    patch_size: int = 14
    grid_size: int = 16
    intermediate_size: int = 4096
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    layer_norm_eps: float = 1e-5
    dtype: str = "float16"

# the very first thing to do is to re-implement the vision-encoder

# the embeddings layer 

# done
class CLIPVisionEmbeddings(Module):
    def __init__(self, config:LlavaConfig):
        # super.__init__() # do we really need this?
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.class_embeddings = Parameter((self.embed_dim,))
        self.patch_embedding = Conv2D(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = Parameter(shape=(self.num_positions,self.embed_dim))

    def forward(self, pixel_values: Tensor) -> Tensor:
        batch_size = pixel_values.shape[0]
        patch_embeds = self.patch_embedding(pixel_values)  # shape = [*, width, grid, grid]
        patch_embeds = reshape(patch_embeds,shape=(batch_size,self.embed_dim,-1))
        patch_embeds = permute_dims(patch_embeds,axes=(0,2,1)) # shape = [batch,grid*grid,embed_dim]
        class_embeds = broadcast_to(self.class_embeddings,shape=(batch_size,1,self.embed_dim)) # shape of (batch,1,embed_dim)
        embeddings  = concat([class_embeds,patch_embeds],dim=1)
        batch_position_embedding = broadcast_to(self.position_embedding,shape=(batch_size,self.num_positions,self.embed_dim))
        embeddings = embeddings +  batch_position_embedding
        return embeddings
    
   

def sigmoid(x: Tensor, name: str = "sigmoid") -> Tensor:
    """Add a new axis to a tensor

    Parameters
    ----------
    x : Tensor
        Input tensor to expand.
    dim : int
        Dimension to expand.
    name : str
        Name hint for this operator.

    Returns
    -------
    result : Tensor
        Expanded result.
    """
    return _wrap_nested(relax.op.sigmoid(x._expr), name)


# done
class LlavaQuickGELU(Module):
    def __init__(self, config: LlavaConfig):
        pass
    def forward(self, input_tensor: Tensor) -> Tensor:
        return input_tensor * sigmoid(input_tensor * 1.702)

# done
class CLIPMLP(Module):
    def __init__(self, config):
        self.activation_fn = LlavaQuickGELU(config=config)
        self.fc1 = Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states

# done
class CLIPAttention(Module):
    def __init__(self,config:LlavaConfig):
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if (self.head_dim * self.num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.k_proj = Linear(self.embed_dim, self.embed_dim)
        self.v_proj = Linear(self.embed_dim, self.embed_dim)
        self.q_proj = Linear(self.embed_dim, self.embed_dim)
        self.out_proj = Linear(self.embed_dim, self.embed_dim)
    
    # return a tensor of shape(batch,num_heads,seq_len,head_dim)
    def _shape(self, tensor: Tensor, seq_len: int, bsz: int):
        reshape_tensor = reshape(tensor,shape=(bsz, seq_len, self.num_heads, self.head_dim))
        permute_tensor = permute_dims(reshape_tensor,axes=(0,2,1,3))
        return permute_tensor
        # return reshape(tensor,shape=(bsz, seq_len, self.num_heads, self.head_dim)).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        causal_attention_mask: Optional[Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) ->Tensor:
        bsz, tgt_len, embed_dim = hidden_states.shape
        query_states = self._shape(self.q_proj(hidden_states) * self.scale,tgt_len,bsz)
        key_states = self._shape(self.k_proj(hidden_states), tgt_len, bsz)
        value_states = self._shape(self.v_proj(hidden_states), tgt_len, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim) # shape of (batch*num_heads, seq_len,head_dim)

        query_states = reshape(query_states,shape=proj_shape)
        key_states = reshape(key_states,shape=proj_shape)
        value_states = reshape(value_states,shape=proj_shape)

        trans_key_states = permute_dims(key_states,axes=(0,2,1))
        attn_weights = matmul(query_states, trans_key_states)

        if attn_weights.shape != [bsz * self.num_heads, tgt_len, tgt_len]:
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, tgt_len)}, but is"
                f" {attn_weights.shape}"
            )

        attn_weights = softmax(attn_weights, axis=-1)
        attn_output = matmul(attn_weights, value_states)
        if attn_output.shape != [bsz * self.num_heads, tgt_len, self.head_dim]:
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.shape}"
            )
        attn_output = reshape(attn_output,shape=(bsz, self.num_heads, tgt_len, self.head_dim))
        attn_output = permute_dims(attn_output,axes=(0,2,1,3))
        attn_output = reshape(attn_output,shape=(bsz,tgt_len,embed_dim))
        attn_output = self.out_proj(attn_output)
        
        return attn_output

#done
class CLIPEncoderLayer(Module):
    def __init__(self, config: LlavaConfig):
        self.embed_dim = config.hidden_size
        self.self_attn = CLIPAttention(config)
        self.layer_norm1 = LayerNorm(normalized_shape=self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = CLIPMLP(config)
        self.layer_norm2 = LayerNorm(normalized_shape=self.embed_dim, eps=config.layer_norm_eps)
    
    def forward(
        self,
        hidden_states: Tensor,
    ):
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states=hidden_states)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        return outputs
    
    
        
# done
class CLIPEncoder(Module):
    def __init__(self, config:LlavaConfig):
        self.layers = ModuleList([CLIPEncoderLayer(config) for _ in range(config.num_hidden_layers)])
    def forward(self,inputs_embeds):
        hidden_states = inputs_embeds
        encoder_states=()
        for idx,encoder_layer in enumerate(self.layers):
            encoder_states = encoder_states + (hidden_states,)
            layer_outputs = encoder_layer(
                hidden_states
            )
            hidden_states = layer_outputs[0]
        encoder_states = encoder_states + (hidden_states,)
        return encoder_states
    
    
        


class CLIPVisionTransformer(Module):
    def __init__(self, config: LlavaConfig):
        embed_dim = config.hidden_size
        self.embeddings = CLIPVisionEmbeddings(config)
        self.pre_layrnorm = LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.encoder = CLIPEncoder(config)

        # Even it is not used for output, it still need to be here for paramaters importing
        self.post_layernorm = LayerNorm(embed_dim, eps=config.layer_norm_eps) 

    def forward(self,pixel_values):
        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.pre_layrnorm(hidden_states)
        encoder_outputs = self.encoder(inputs_embeds=hidden_states)
        return encoder_outputs

class CLIPVisionModel(Module):
    def __init__(self, config: LlavaConfig):
        self.vision_model = CLIPVisionTransformer(config)
    def forward(self,pixed_values):
        return self.vision_model(pixed_values)
    
    def test(self):
        import numpy as np
        x = Tensor.from_const(np.zeros(shape=(1,3,224,224)))
        y = self.forward(x)[-2]
        print("===========",y)
        return y

    





clip = CLIPVisionModel(LlavaConfig)

def test_tensor_op_binary_tensor_tensor():
    class Model(Module):
        def test(self, x: Tensor, y: Tensor):
            z0 = x + y
            z1 = x * y
            z2 = x / y
            z3 = x.maximum(y)
            z4 = x.minimum(y)
            print(z2)
            return (z0, z1, z2, z3, z4)

    # fmt: off
    @R.function
    def test(x: R.Tensor((1, 10), dtype="float32"), y: R.Tensor((2, 1), dtype="float32"), _io: R.Object) -> R.Tuple(R.Tuple(R.Tensor((2, 10), dtype="float32"), R.Tensor((2, 10), dtype="float32"), R.Tensor((2, 10), dtype="float32"), R.Tensor((2, 10), dtype="float32"), R.Tensor((2, 10), dtype="float32")), R.Tuple(R.Object)):
        R.func_attr({"num_input": 3})
        with R.dataflow():
            add: R.Tensor((2, 10), dtype="float32") = R.add(x, y)
            mul: R.Tensor((2, 10), dtype="float32") = R.multiply(x, y)
            divide: R.Tensor((2, 10), dtype="float32") = R.divide(x, y)
            maximum: R.Tensor((2, 10), dtype="float32") = R.maximum(x, y)
            minimum: R.Tensor((2, 10), dtype="float32") = R.minimum(x, y)
            gv1: R.Tuple(R.Tuple(R.Tensor((2, 10), dtype="float32"), R.Tensor((2, 10), dtype="float32"), R.Tensor((2, 10), dtype="float32"), R.Tensor((2, 10), dtype="float32"), R.Tensor((2, 10), dtype="float32")), R.Tuple(R.Object)) = (add, mul, divide, maximum, minimum), (_io,)
            R.output(gv1)
        return gv1
    # fmt: on

    m = Model()
    irmodule, _ = m.export_tvm(
        spec={"test": {"x": spec.Tensor([1, 10], "float32"), "y": spec.Tensor([2, 1], "float32")}},
        debug=True,
    )

    tvm.ir.assert_structural_equal(irmodule["test"], test)

# test_tensor_op_binary_tensor_tensor()
irmodule,_ = clip.export_tvm(
    spec={"test":{}},
    debug=False
)


# the pre_norm layer

# the encder layer

# the post_norm layer



