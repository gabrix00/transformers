import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
from transformers.models.bert.modeling_bert import BertSelfAttention, BertEmbeddings, BertEncoder, BertPooler

# Define your custom self-attention layer
class CustomSelfAttention(nn.Module):
    def __init__(self, config):
        super(CustomSelfAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None,
                encoder_attention_mask=None, past_key_value=None, output_attentions=False, gabriel_mask=None):
        self_output = self.self(hidden_states, attention_mask, head_mask, encoder_hidden_states,
                                encoder_attention_mask, past_key_value, output_attentions, gabriel_mask)
        return self.dropout(self_output[0])

# Create a modified BERT model with the custom attention layer
class ModifiedBertModel(nn.Module):
    def __init__(self, config):
        super(ModifiedBertModel, self).__init__()
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.attention = CustomSelfAttention(config)

    def forward(self, input_ids, attention_mask=None, head_mask=None, encoder_hidden_states=None,
                encoder_attention_mask=None, past_key_value=None, output_attentions=False, gabriel_mask=None):
        embedding_output = self.embeddings(input_ids)
        encoder_outputs = self.encoder(embedding_output, attention_mask, head_mask, encoder_hidden_states,
                                       encoder_attention_mask, past_key_value, output_attentions)
        attention_output = self.attention(encoder_outputs[0], attention_mask, head_mask, encoder_hidden_states,
                                          encoder_attention_mask, past_key_value, output_attentions, gabriel_mask)
        pooled_output = self.pooler(attention_output[:, 0])
        return pooled_output, encoder_outputs

# Create the BertClassifier class using the modified BERT model
class BertClassifier(nn.Module):
    """Bert Model for Classification Tasks.
    """
    def __init__(self, config):
        super(BertClassifier, self).__init__()
        self.bert = ModifiedBertModel(config)
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, 50),
            nn.ReLU(),
            nn.Linear(50, 5)
        )

    def forward(self, input_ids, attention_mask=None, head_mask=None, encoder_hidden_states=None,
                encoder_attention_mask=None, past_key_value=None, output_attentions=False, gabriel_mask=None):
        pooled_output, _ = self.bert(input_ids, attention_mask, head_mask, encoder_hidden_states,
                                     encoder_attention_mask, past_key_value, output_attentions, gabriel_mask)
        logits = self.classifier(pooled_output)
        return logits

# Example usage
config = BertConfig.from_pretrained('bert-base-uncased')
config.vocab_size = 30522  # Update vocab size if necessary
b = BertClassifier(config)
token_ids = torch.tensor([[1, 2, 3, 4]])  # Example input token IDs
attention_mask = torch.tensor([[1, 1, 1, 1]])  # Example attention mask
gabriel_mask = torch.tensor([[[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]], dtype=torch.float32)  # Example gabriel mask
out = b(input_ids=token_ids, attention_mask=attention_mask, gabriel_mask=gabriel_mask)
print(out)
