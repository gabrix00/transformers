import matplotlib.pyplot as plt
import numpy as np
import torch 
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def from_parser2masking(text:str, list_of_mapped_rel:list, viz:bool = False):

    encodes = tokenizer(text,add_special_tokens=False,return_tensors='pt')
    encodes = encodes['input_ids'][0].tolist()
    decodes = [tokenizer.decode(enc) for enc in encodes]

    n = len(decodes)

    mask = np.zeros((n+2, n+2)) #+2 per [CLS] e [SEP] token

    # adjusted_decodes serve a mappare correttamente i decodes, evitando il mismatch tra parole splittate es token_2, ##izer_2
    adjusted_decodes = []
    index_adjustment = 0
    for index, token in enumerate(decodes):
            if token[:2]=='##':
                index_adjustment += 1
                adjusted_decodes.append(str(token)+'_'+str(index - index_adjustment))
                
            else:
                adjusted_decodes.append(str(token)+'_'+str(index - index_adjustment))

    adjusted_decodes = ['[CLS]'] + adjusted_decodes + ['[SEP]']

    #print(adjusted_decodes) #debug
    for i, token in enumerate(adjusted_decodes):
        for j, other_token in enumerate(adjusted_decodes):
            if i != j:
                #print(token,other_token) #debug
                if token == '[CLS]' or token == '[SEP]' :
                    continue
                if (token,other_token) in list_of_mapped_rel:
                    mask[i][j] = 1
                elif other_token[:2]=='##' and (token.split('_')[1] == other_token.split('_')[1]) and (token != '[CLS]' or  token != '[SEP]'): #attendo stessa parola splittata es token_2, ##izer_2
                    mask[i][j] = 1
                    mask[j][i] = 1  # attendere sia token -->##izer in quanto stessa parola, ma anche ##izer-->token per completezza
            else:   
                mask[i][i] = 1  #caso token attende con se stesso a priori


                
    if viz:
        plt.imshow(mask, cmap='Blues', interpolation='nearest')
        plt.title('Attention Mask')

        adjusted_decodes = []
        index_adjustment = 0 
        for index, token in enumerate(decodes):
            if token[:2]=='##':
                index_adjustment += 1 #ogni volta che si incontra un token con ## davanti si scala il suo indice e di tutti token successivi di 1
                adjusted_decodes.append(str(token)+'_'+str(index - index_adjustment))
                
            else:
                adjusted_decodes.append(str(token)+'_'+str(index - index_adjustment))
        
        adjusted_decodes = ['[CLS]'] + adjusted_decodes + ['[SEP]']
            
        plt.xticks(range(len([el for el in adjusted_decodes])), adjusted_decodes,rotation=45)
        plt.yticks(range(len([el for el in adjusted_decodes])), adjusted_decodes,rotation=45)
        plt.colorbar()
        plt.show()

    #print(mask)
    return torch.tensor(mask)

##### PER TAMPONARE ######

#from spacy_dependency import create_dependency_pairs
#text = "A BERT tokenizer uses something known BERT tokenizer which is BERT case sensitive"
#text = "BERT tokenizer uses something known as subword-based tokenization. Subword-tokenization splits unknown words into smaller words or characters such that the model can derive some meaning from the tokens."
#text = 'A black hole is a region of spacetime where gravity is so strong that nothing'
#create_dependency_pairs(text)

#print(create_dependency_pairs(text))

#from_parser2masking(text,create_dependency_pairs(text),True)

