import spacy
#from transformers import BertTokenizer
from transformers import AutoTokenizer

nlp = spacy.load("en_core_web_sm")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

#text = "A BERT tokenizer uses something known BERT tokenizer which is BERT case sensitive"

def mapping (tokens_list:list, dict_to_update:dict):
    maps = {}

    #processo che mi fa i songoli encoding dei token per cui vi è un miosmatch tra i due tokenizer
    for t in tokens_list:
        encodes = tokenizer(t,add_special_tokens=False,return_tensors='pt')
        encodes = encodes['input_ids'][0].tolist()
        decodes = [tokenizer.decode(enc) for enc in encodes]


        if t not in maps: # creo il dict che presenta solo i mismatch chiave tokenizer spacy, valore tokenizer bert
            maps[t] = decodes # lo riempio, questo dict non avrà gli idnci di posizone! perchè t è proviene dalla lista 'spacy_tokenizzation'! es: tokenizer:[token, ##izer]
    
    for k in dict_to_update: # dict_to_update è il dict di mappe chiave valore (chiave spacy, valore spacy), lo voglio aggiornare in modo tale che dove è presente un mismatch tra i due tokenixer, abbia chiave spacy, valore pythorch
        if str(k).split('_')[0] in maps: # str(k).split('_')[0] mi serve ad epurare il token dal suffisso numerico, perchè la mappa in pythorch non lo presenta (infatti è tipo tokenizer_3:[token,##izer])
            dict_to_update[k] = maps[str(k).split('_')[0]] #aggiorno il dict con la mappa corretta (in questo caso per tutti i token per cui c'è un mismatch tra i due tokenizer)
 
    return dict_to_update




def spacy_map(text:str):
    
    #tokenizer Spacy and Bert

    sentence = nlp(text)
    #tokens = tokenizer(text,add_special_tokens=False,return_tensors='pt',return_offsets_mapping=True) #NotImplementedError: return_offset_mapping is not available when using Python tokenizers. To use this feature, change your tokenizer to one deriving from transformers.PreTrainedTokenizerFast
    tokens = tokenizer(text,add_special_tokens=False,return_tensors='pt',return_offsets_mapping=True) #correct


    spacy_tokenizzation = [token.text.lower() for token in sentence]
    bert_tokenizzation  = [tokenizer.decode(token) for token in tokens['input_ids'][0].tolist()]


    tokens_mismatch = [token for token in spacy_tokenizzation if token not in bert_tokenizzation] # prendo la lista dei mismatch tra i due tokenizer

    #dict to pass to the mapping function che si chiamerà dict_to_update in 'mapping'
    spacy_tokenizzation_dict = dict([(token.text.lower()+'_'+str(index),token.text.lower()+'_'+str(index)) for index,token in enumerate(sentence)]) #mi creo le mappa con gli indici di posizione

    return mapping(tokens_mismatch,spacy_tokenizzation_dict)


if __name__ == '__main__':
    print(spacy_map(text))


