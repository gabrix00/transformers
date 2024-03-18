from distutils import config
import torch
from torch.utils.data import DataLoader
from torch import device
from torch import nn
from typing import List, Optional, Tuple, Union
import math
from tqdm import tqdm 
from transformers import AutoTokenizer
from masking_process import masking
#from transformers.models.bert.modeling_bert import BertModel
from transformers import BertModel
#from transformers.models.bert.modeling_bert import BertSelfAttention
from transformers.models.bert.modeling_bert import BertSelfAttention, BertConfig
#model = BertModel.from_pretrained('bert-base-uncased')
# Creare un'istanza del tuo modello





"""
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
        gabriel_mask=None,  # Aggiunto gabriel_mask come parametro
    ):
        # Chiama il metodo forward di BertSelfAttention
        outputs = super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
        )

        # Estrai attention_probs dagli outputs
        attention_probs = outputs[1]

        # Applica gabriel_mask se fornito
        if gabriel_mask is not None:
            attention_probs *= gabriel_mask

        # Aggiorna gli outputs con attention_probs modificati
        outputs = (outputs[0], attention_probs) + outputs[2:]

        return outputs
"""

class CustomBertSelfAttention(nn.Module):
    def __init__(self, gabriel_mask=None):
        super().__init__()
        # La tua implementazione del costruttore
        self.gabriel_mask = gabriel_mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
        gabriel_mask: Optional[torch.FloatTensor] = None,
    ) -> Tuple[torch.Tensor]:
        if gabriel_mask is not None:
            self.gabriel_mask = gabriel_mask

        mixed_query_layer = self.query(hidden_states)
        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        use_cache = past_key_value is not None
        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            if use_cache:
                position_ids_l = torch.tensor(key_length - 1, dtype=torch.long, device=hidden_states.device).view(
                    -1, 1
                )
            else:
                position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        
        # modifica gabriel_mask
        if gabriel_mask is not None:
            print('there is a gabriel mask!')
            #torch.addcmul(0,gabriel_mask,attention_scores,1)
            attention_scores *= gabriel_mask
            print(attention_scores)



        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs
        

"""
class BertSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None, gabriel_mask = None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder
        #self.gabriel_mask = config.gabriel_mask
        self.gabriel_mask = gabriel_mask

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
        gabriel_mask: Optional[torch.FloatTensor] = None,
    ) -> Tuple[torch.Tensor]:
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        use_cache = past_key_value is not None
        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            if use_cache:
                position_ids_l = torch.tensor(key_length - 1, dtype=torch.long, device=hidden_states.device).view(
                    -1, 1
                )
            else:
                position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # modifica gabriel_mask
        if gabriel_mask is not None:
            print('there is a gabriel mask!')
            #torch.addcmul(0,gabriel_mask,attention_scores,1)
            attention_scores *= gabriel_mask
            print(attention_scores)



        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs
"""
"""
class CustomSelfAttention(nn.Module):
    def __init__(self, config):
        super(CustomSelfAttention, self).__init__()

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None,
                encoder_attention_mask=None, past_key_value=None, output_attentions=False, gabriel_mask=None):
        self_output = self.self(hidden_states, attention_mask, head_mask, encoder_hidden_states,
                                encoder_attention_mask, past_key_value, output_attentions, gabriel_mask)
        return self.dropout(self_output[0])
"""
"""
class CustomBERTModel(nn.Module):
    def __init__(self):
        super(CustomBERTModel, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        #self.gabriel_mask = gabriel_mask

    def forward(self, input_ids, attention_mask, gabriel_mask):#token_type_ids
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            #token_type_ids=token_type_ids,
            gabriel_mask=gabriel_mask  # Passa gabriel_mask qui
        )
        # Puoi manipolare ulteriormente gli output se necessario
        return outputs
"""
from transformers import BertModel

class CustomBERTModel(nn.Module):
    def __init__(self, config,gabriel_mask=None):
        super(CustomBERTModel, self).__init__()
        self.bert = BertModel(config)
        self.gabriel_mask = gabriel_mask 

    def forward(self, input_ids, attention_mask, token_type_ids=None, gabriel_mask=None):
        # Rimuovi gli argomenti non validi prima di chiamare il metodo forward() del modello BERT
        #if gabriel_mask is not None:
            #print(gabriel_mask)
            #print('RIMOSSO GABRIEL MASK!')
            #del gabriel_mask
        print('GABRIEL MASK!')
        print(gabriel_mask)

        self.bert.encoder.layer[0].attention.self = CustomBertSelfAttention(gabriel_mask=self.gabriel_mask)


        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        # Puoi manipolare ulteriormente gli output se necessario
        return outputs

"""
class CustomBERTModel(BertModel):
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
        gabriel_mask=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            gabriel_mask=gabriel_mask,  # Passa gabriel_mask qui
        )

        return encoder_outputs
"""
"""
class CustomBERTModel(BertModel):
    def forward(self, input_ids, attention_mask, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, encoder_hidden_states=None, encoder_attention_mask=None, output_attentions=None, output_hidden_states=None, return_dict=None, gabriel_mask=None):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, device)

        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            gabriel_mask=gabriel_mask,  # Passa gabriel_mask qui
        )

        return encoder_outputs
"""
class Dataset:
    def __init__(self, texts, tokenizer, max_len):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        inputs = self.tokenizer.__call__(text,
                                        None,
                                        add_special_tokens=True,
                                        max_length=self.max_len,
                                        padding="max_length",
                                        truncation=True,
                                        )
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        gabriel_mask = masking(text)
        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "gabriel_mask": torch.tensor(gabriel_mask, dtype=torch.long)
        }
#"A BERT tokenizer uses something known BERT tokenizer which is BERT case sensitive"
def main():
    # Creare CustomBERTModel
    config = BertConfig.from_pretrained("bert-base-uncased")
    model = CustomBERTModel(config)

    # Stampare i nomi dei moduli dopo la creazione di CustomBERTModel
    #print("Moduli dopo la creazione di CustomBERTModel:")
    #print([name for name, _ in model.named_modules()])

    
    #print('++++++++++++++++++++++')
    #model = CustomBERTModel()
    #print('++++++++++++++++++++++')

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    texts = ["A BERT tokenizer"]#,"A BERT tokenizer uses something known BERT"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()  # Imposta il modello in modalità valutazione
    train_dataset = Dataset(texts=texts, tokenizer=tokenizer, max_len=6)	#Dataset prende texts (si suppone essere una lista di stringhe)
    print('\n\n')
    # Stampa degli elementi del dataset
    print(train_dataset[0]) #ids,padding/mask,gabriel_mask
    print('------')

    train_data_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=2)
    print('\n\n')
    print(len(train_data_loader))
    print('------')

    
    

    for bi, d in tqdm(enumerate(train_data_loader), total=len(train_data_loader), desc='Loading:', disable=True):
        ids = d["ids"].to(device, dtype=torch.long)#,non_blocking=True)
        mask = d["mask"].to(device, dtype=torch.long)#,non_blocking=True)
        gabriel_mask = d['gabriel_mask'].to(device, dtype=torch.long)#,non_blocking=True)
        outputs = model(input_ids=ids, attention_mask=mask, gabriel_mask=gabriel_mask) # aggiungere modifiche
        print(outputs)
        #print(ids)
        #print(mask)
        #print(gabriel_mask)

if __name__ == '__main__':
    main()


# Eseguire il forward pass
#outputs = custom_bert_model(input_ids, attention_mask, token_type_ids, gabriel_mask)
