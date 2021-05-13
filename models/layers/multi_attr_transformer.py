import math
import torch
import torch.nn as nn
from models.layers.bilinear_layer import Bilinear
from transformers.activations import gelu, gelu_new, silu

def mish(x):
    return x * torch.tanh(nn.functional.softplus(x))
BertLayerNorm = torch.nn.LayerNorm
ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": silu, "gelu_new": gelu_new, "mish": mish}


class AttributeAttention(nn.Module):
    def __init__(self, config, cus_config):
        super().__init__()
        if cus_config.attr_dim % cus_config.num_attr_heads != 0:
            raise ValueError(
                "The attribute hidden size (%d) is not a multiple of the number of attribute heads "
                "heads (%d)" % (cus_config.attr_dim, config.num_attr_heads)
            )

        self.att_type = cus_config.type

        self.num_attr_heads = cus_config.num_attr_heads
        self.attention_head_size = int(cus_config.attr_dim / cus_config.num_attr_heads)
        self.all_head_size = self.num_attr_heads * self.attention_head_size

        assert self.att_type in ["a","c", "d"], print("error att_type")
        if self.att_type == 'a': # a,c,d
            self.query = nn.Linear(config.hidden_size, self.all_head_size)
            self.key = nn.Linear(config.hidden_size, self.all_head_size)
            self.value = nn.Linear(config.hidden_size, self.all_head_size)

        if self.att_type == 'c': # a,c,d
            self.query = Bilinear(config.hidden_size, self.all_head_size)
            self.key = Bilinear(config.hidden_size, self.all_head_size)
            self.value = nn.Linear(config.hidden_size, self.all_head_size)

        if self.att_type == 'd': # a,c,d
            self.query = Bilinear(config.hidden_size, self.all_head_size)
            self.key = Bilinear(config.hidden_size, self.all_head_size)
            self.value = Bilinear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(cus_config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attr_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        attr,
        hidden_states,
        attention_mask=None,
        head_mask=None,
    ):
        if self.att_type == 'a':
            mixed_query_layer = self.query(hidden_states)
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)
        elif self.att_type == 'c':
            mixed_query_layer = self.query(hidden_states, attr)
            mixed_key_layer = self.key(hidden_states, attr)
            mixed_value_layer = self.value(hidden_states)
        else:
            mixed_query_layer = self.query(hidden_states, attr)
            mixed_key_layer = self.key(hidden_states, attr)
            mixed_value_layer = self.value(hidden_states, attr)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return context_layer

class BertSelfOutput(nn.Module):
    def __init__(self, cus_config):
        super().__init__()
        self.dense = nn.ModuleList([nn.Linear(cus_config.attr_dim, cus_config.attr_dim) for _ in range(cus_config.num_attrs)])
        self.project = nn.Linear(cus_config.attr_dim*cus_config.num_attrs, cus_config.attr_dim)
        self.all_dense = nn.Linear(cus_config.attr_dim*cus_config.num_attrs, cus_config.attr_dim)
        self.LayerNorm = BertLayerNorm(cus_config.attr_dim, eps=cus_config.layer_norm_eps)
        self.dropout = nn.Dropout(cus_config.hidden_dropout_prob)

    def forward(self, input_tensor, *hidden_states):
        # v1
        new_hidden_states = []
        for hidden_state, dense in zip(hidden_states, self.dense):
            new_hidden_states.append(self.dropout(dense(hidden_state)))
        # # hidden_states = torch.stack(new_hidden_states, 0).sum(0)
        # v2
        hidden_states = self.project(torch.cat(new_hidden_states, dim=-1))

        # v3
        # hidden_states = self.project(torch.cat(hidden_states, dim=-1))

        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class Attention(nn.Module):
    def __init__(self, config, cus_config):
        super().__init__()
        self.atts = nn.ModuleList([AttributeAttention(config, cus_config) for _ in range(cus_config.num_attrs)])
        self.output = BertSelfOutput(cus_config)

    def forward(self, attrs, embeddings, mask):
        hidden_states = self.output(embeddings,
                                    *[att(attr=attr, hidden_states=embeddings, attention_mask=mask)
                                      for attr, att in zip(attrs, self.atts)]
                                    )
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, cus_config):
        super().__init__()
        self.dense = nn.Linear(cus_config.intermediate_size, cus_config.attr_dim)
        self.LayerNorm = BertLayerNorm(cus_config.attr_dim, eps=cus_config.layer_norm_eps)
        self.dropout = nn.Dropout(cus_config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertIntermediate(nn.Module):
    def __init__(self, cus_config):
        super().__init__()
        self.dense = nn.Linear(cus_config.attr_dim, cus_config.intermediate_size)
        if isinstance(cus_config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[cus_config.hidden_act]
        else:
            self.intermediate_act_fn = cus_config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class MAALayer(nn.Module):
    def __init__(self, config, cus_config):
        super().__init__()
        self.attention = Attention(config, cus_config)
        self.intermediate = BertIntermediate(cus_config)
        self.output = BertOutput(cus_config)

    def forward(
        self,
        attrs,
        hidden_states,
        mask=None,
    ):
        attention_output = self.attention(attrs, hidden_states, mask)
        intermediate_output = self.intermediate(attention_output)
        outputs = self.output(intermediate_output, attention_output)
        return outputs
