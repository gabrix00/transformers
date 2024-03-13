from SpacyDep import dpairs
from Parser2Masking import from_parser2masking



#text = "BERT tokenizer uses something known as subword-based tokenization. Subword-tokenization splits unknown words into smaller words or characters such that the model can derive some meaning from the tokens."
#text = "A BERT tokenizer uses something known BERT tokenizer which is BERT case sensitive"

#create_dependency_pairs(text)

#print(create_dependency_pairs(text))
def masking(text):
    return from_parser2masking(text,dpairs(text),False)
    

#print(masking(text))