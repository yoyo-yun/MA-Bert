import torch
import torch.nn as nn
import torch.nn.init as init
from transformers import BertModel, BertPreTrainedModel, BertLayer
from models.layers.multi_attr_transformer import MAALayer
from models.layers.classifier import BERTClassificationHead, BERTClassificationHeadWithAttribute
from models.layers.fusion_layer import Fusion

class MAAModel(BertPreTrainedModel):
    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.config = config
        self.num_labels = config.num_labels
        self.cus_config = kwargs['cus_config']
        self.type = self.cus_config.type # a,b,c,d

        self.usr_embed = nn.Embedding(self.cus_config.num_usrs, self.cus_config.attr_dim)
        self.usr_embed.weight.requires_grad = True
        init.uniform_(self.usr_embed.weight, a=-0.25, b=0.25)

        self.prd_embed = nn.Embedding(self.cus_config.num_prds, self.cus_config.attr_dim)
        self.prd_embed.weight.requires_grad = True
        init.uniform_(self.usr_embed.weight, a=-0.25, b=0.25)

        if self.type not in ['b', 'a']:
            self.text = nn.Parameter(torch.Tensor(1, self.cus_config.attr_dim))
            # init.normal_(self.text)
            init.uniform_(self.text, a=-0.25, b=0.25)
            self.ATrans_decoder = nn.ModuleList([MAALayer(config, self.cus_config) for _ in range(self.cus_config.n_mmalayer)])
            self.classifier = BERTClassificationHead(config)
        elif self.type == 'a':
            self.fusion = Fusion(self.config.hidden_size,self.cus_config.attr_dim)
            self.layer = nn.ModuleList([BertLayer(config) for _ in range(self.cus_config.n_mmalayer)])
            self.classifier = BERTClassificationHead(config)
        else:
            self.classifier = BERTClassificationHeadWithAttribute(self.cus_config)

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attrs=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=True,
            output_hidden_states=True
        )
        last_output = outputs[0] # last hidden_state in BERT
        pool_output = outputs[1] # pooled hidden_state over [CLS] in the last layer
        all_hidden_states, all_attentions = outputs[2:] # (bs,)

        usrs, prds = attrs # (bs, ) * 2
        usr = self.usr_embed(usrs) # (bs, attr_dim)
        prd = self.prd_embed(prds) # (bs, attr_dim)
        if self.type == 'b':
            hidden_state = self.dropout(last_output)
            outputs = self.classifier(hidden_state, [usr, prd])
        elif self.type == 'a':
            extend_attention_mask = self.get_attention_mask(attention_mask)
            if 12 > self.cus_config.n_bertlayer > 0:
                last_output = all_hidden_states[-(self.config.num_hidden_layers + 1 -self.cus_config.n_bertlayer)]
            hidden_state = self.fusion(last_output, [usr, prd])
            hidden_state = self.dropout(hidden_state)
            for i, l in enumerate(self.layer):
                hidden_state = l(hidden_state, extend_attention_mask)[0]
            hidden_state = self.dropout(hidden_state)
            outputs = self.classifier(hidden_state)
        else:
            t_self = self.text.expand_as(usr)  # (bs, attr_dim)
            extend_attention_mask = self.get_attention_mask(attention_mask)
            if 12 > self.cus_config.n_bertlayer > 0:
                last_output = all_hidden_states[-(self.config.num_hidden_layers + 1 -self.cus_config.n_bertlayer)]
            hidden_state = self.dropout(last_output)
            for i, mmalayer in enumerate(self.ATrans_decoder):
                hidden_state = mmalayer([usr, prd, t_self], hidden_state, extend_attention_mask)
            hidden_state = self.dropout(hidden_state)
            outputs = self.classifier(hidden_state)

        return (outputs, hidden_state)  # logits, last_hidden_stat

    def get_attention_mask(self, attention_mask):
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for input_ids or attention_mask"
                )
        try:
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        except:
            print(extended_attention_mask)
            exit()
        # extended_attention_mask = ~extended_attention_mask * -10000.0
        return extended_attention_mask