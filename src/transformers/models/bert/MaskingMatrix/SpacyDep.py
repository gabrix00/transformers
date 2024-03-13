import spacy
from SpacyMap import spacy_map
from transformers import AutoTokenizer

nlp = spacy.load("en_core_web_sm")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
#text = "A BERT tokenizer uses something known BERT tokenizer which is BERT case sensitive"




def spacy_dep(text:str):
    sentence = nlp(text)
    tokens_children_dep = []
    for index,token in enumerate(sentence):
        for child in token.children:
            tokens_children_dep.append((token.text.lower()+'_'+str(index), # mi creo tutte le coppie tenendo traccia dei relativi suffissi di posizione 
                                        child.text.lower()+'_'+str(list(sentence).index(child))))
            # str(list(sentence).index(child) mi permette di aggiungere il suffisso numerico associato all'indice di posizone di child in sentence.
            # ossia è come se stessi andando a prendere l'indice di posizione di un token in sentence (lista di token spacy), infatti child è un token!
        
    return tokens_children_dep


def dpairs(text:str):
    lista_dipendenze = spacy_dep(text)

    lista_dipendenze_mappate = [] #secondo il tokenizer di bert
    for tup in lista_dipendenze:
        suffix_f= '_'+str(tup[0]).split('_')[1]
        suffix_s= '_'+str(tup[1]).split('_')[1]

        # Controllo se entrambi i termini della tupla sono liste
        if isinstance(spacy_map(text)[tup[0]], list) and isinstance(spacy_map(text)[tup[1]], list):
            # Aggiungi i suffissi appropriati a ogni elemento delle liste
            updated_left_list = [el + suffix_f for el in spacy_map(text)[tup[0]]]
            updated_right_list = [el + suffix_s for el in spacy_map(text)[tup[1]]]
            # Aggiungi le coppie di liste modificate alla lista delle dipendenze mappate
            lista_dipendenze_mappate.append((updated_left_list, updated_right_list))
            continue

        # Controllo se solo il primo termine della tupla è una lista
        if isinstance(spacy_map(text)[tup[0]], list):
        # Se il valore è una lista, aggiungi il suffisso a ogni elemento
            updated_left_list = [el + suffix_f for el in spacy_map(text)[tup[0]]]
            lista_dipendenze_mappate.append((updated_left_list,spacy_map(text)[tup[1]]))
            continue

        # Controllo se solo il secondo termine della tupla è una lista
        if isinstance(spacy_map(text)[tup[1]], list):
            updated_right_list = [el + suffix_s for el in spacy_map(text)[tup[1]]]
            lista_dipendenze_mappate.append((spacy_map(text)[tup[0]],updated_right_list))
            continue
  
        lista_dipendenze_mappate.append((spacy_map(text)[tup[0]],spacy_map(text)[tup[1]])) #nel caso nessuno dei due sia lista si ottengono le mappe normali


    ##### !!!CASISTICA!!! ##### 
    update_lista_dipendenze_mappate=[]
    for tup in lista_dipendenze_mappate:
        # caso prima elemento della tuple è una lista e il secondo è un stringa es: (['token', '##izer'], 'bert')
        if type(tup[0]) == list and type(tup[1]) == str:
            for el in tup[0]:
                update_lista_dipendenze_mappate.append((el,tup[1]))

        # caso prima elemento della tuple è una lista e il secondo è una lista  es: (['token', '##ization'], ['sub', '##word'])
        elif type(tup[0]) == list and type(tup[1]) == list:
            for el1 in tup[0]:
                for el2 in tup[1]:
                    update_lista_dipendenze_mappate.append((el1,el2))

        # caso primo elemento della tupla è una stringa e il secondo è una lista es: ('uses', ['token', '##izer'])
        elif type(tup[0]) == str and type(tup[1]) == list:
            for el in tup[1]:
                    update_lista_dipendenze_mappate.append((tup[0],el))

        # caso primo elemento della tupla è una stringa e il secondo è una stringa es: ('uses', 'something')
        else:
            update_lista_dipendenze_mappate.append(tup) #nessuna modifica

    return update_lista_dipendenze_mappate


'''
print('SPACY MAP:')
print(spacy_map(text))
print('\n\n')

print('SPACY DEPENDENCY:')
print(spacy_dependency(text))
print('\n\n')


print('create_dependency_pairs:')
print(create_dependency_pairs(text))
print('\n\n')
'''