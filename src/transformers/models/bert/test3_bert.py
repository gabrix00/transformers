from transformers import BertModel, BertConfig
from transformers.models.bert.modeling_bert import BertSelfAttention

class CustomBertSelfAttention(BertSelfAttention):
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        gabriel_mask=None,  # Custom argument
    ):
        # Your custom self-attention implementation using gabriel_mask
        # ...

class CustomBertModel(BertModel):
    def __init__(self, config):
        super().__init__(config)
        # Replace the self-attention layers with your custom self-attention layer
        for layer in self.encoder.layer:
            layer.attention.self = CustomBertSelfAttention(config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        gabriel_mask=None,  # Custom argument
    ):
        # Pass the custom argument to the self-attention layers
        encoder_outputs = self.encoder(
            encoder_hidden_states,
            attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            gabriel_mask=gabriel_mask,  # Pass the custom argument
        )

        # Continue with the rest of the forward pass
        # ...