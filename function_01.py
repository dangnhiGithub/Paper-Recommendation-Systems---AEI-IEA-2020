# standard
import numpy as np
import pandas as pd

# visualize
from tqdm.notebook import tqdm
from itertools import product

# system
import pickle     ## saving library
import os         ## file manager
import sys
from multiprocessing import Pool
import time

import re         ## preprocessing text library
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')     # download toolkit for textblob.TextBlob.words

from textblob import TextBlob
from nltk.stem import PorterStemmer     # tranform expanding words of words like attacker, attacked, attacking -> attack
st = PorterStemmer()

stop_words = stopwords.words('english')
stop = ["a", "about", "above", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along", "already", "also","although","always","am","among", "amongst", "amoungst", "amount",  "an", "and", "another", "any","anyhow","anyone","anything","anyway", "anywhere", "are", "around", "as",  "at", "back","be","became", "because","become","becomes", "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom","but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either", "eleven","else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "isnt", "it", "its", "itself", "keep", "keeps", "kept", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "names", "named", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "ok", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own","part", "per", "perhaps", "please", "put", "puts", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "sees", "serious", "several", "she", "should", "show", "shows", "showed", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "thickv", "thin", "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves", "the", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]


class preprocess_text():
    def preprocess_Abstract(self, text):
        if type(text) == str:
            text = text.lower()
            text = re.sub("[\d+]", "", text)
            text = re.sub("[^a-z]", " ", text)

            filtered_sentence = [] 
            for w in text.split(" "): 
                if w not in stop_words and w not in stop:
                    filtered_sentence.append(w)
            text = " ".join(filtered_sentence)
            text = text.strip()
            text = re.sub("[ ]{2,}", " ", text)
        return text

    def preprocess_Title(self, text):
        return self.preprocess_Abstract(text)

    def preprocess_Aims(self, text):
        return self.preprocess_Abstract(text)
    
    def preprocess_Keywords(self, text):
        return self.preprocess_Abstract(text)

class preprocess_tool():
    def __init__(self, tool_preprocessText = preprocess_text()):
        self.tool_preprocessText = tool_preprocessText

    def get_preprocessed_data(self, dataframe, 
                              preprocess_columns = ['title', 'abstract', 'keywords'],
                              preprocessing_type = ['Title', 'Abstract', 'Keywords'],
                              keep_columns = ['itr'],
                              n_jobs=4):
        '''-Parameters:
              preprocess_columns: choosen columns to apply preprocessing method have definded in class preprocess_text() format.
              preprocessing_type: preprocessing methods are applied to respective preprocessing columns have definded in class preprocess_text() format.
              keep_columns: columns which we do nothing.
           -Return:
              (pandas DataFrame): data after preprocess
        '''########
        output_data = pd.DataFrame(columns=preprocessing_type + keep_columns)
        output_data[preprocessing_type + keep_columns] = dataframe[preprocess_columns + keep_columns]

        if 'Title' in preprocessing_type:
            with Pool(n_jobs) as p:
                output_data['Title'] = p.map(self.tool_preprocessText.preprocess_Title, output_data['Title'])
        if 'Abstract' in preprocessing_type:
            with Pool(n_jobs) as p:
                output_data['Abstract'] = p.map(self.tool_preprocessText.preprocess_Abstract, output_data['Abstract'])
        if 'Keywords' in preprocessing_type:
            with Pool(n_jobs) as p:
                output_data['Keywords'] = p.map(self.tool_preprocessText.preprocess_Keywords, output_data['Keywords'])
        if "Aims" in preprocessing_type:
            with Pool(n_jobs) as p:
                output_data['Aims'] = p.map(self.tool_preprocessText.preprocess_Aims, output_data['Aims'])
        return output_data


def labelling_data(series, category):
    '''-Parameter:
          series(pandas Series): Conference distribution of data.
          category(Int64Index list): category (do not reset_index of aims_content before using this funtion)
        -Return: 
          (np array): label series for data.
    '''########
    label = np.zeros(len(series))
    for i, j in enumerate(category):
        label[series == j] = i
    return label.astype(int)



from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import io

class NLP_tool():
    def __init__(self):
        pass

    def build_tokenizer(self, contents, vocabulary_size = None):
        '''-Parameters:
              contents(list): list of texts or strings.
           -Return: tokenizer object.
        '''######
        tokenizer = Tokenizer(num_words = vocabulary_size)
        tokenizer.fit_on_texts(contents)
        return tokenizer

    def tokenize_data(self, Series, tokenizer, maxlen=None):
        '''- Parameters:
              Series(pandas DataFrame Series): list of strings
              tokenizer(keras Tokenizer)
              maxlen(int): max length tokenkize
           - Return: 
              (numpy array): tokenized matrix of input Series
        '''######
        
        contents = Series.tolist()
        sequences = tokenizer.texts_to_sequences(contents)
        if maxlen == None:
            data = pad_sequences(sequences)
        else:
            data = pad_sequences(sequences, maxlen=maxlen)
        return data

    def download_FastText_pretrained(self):
        if os.path.exists('crawl-300d-2M.vec'):
            print("File have already downloaded.")
            return
        os.system('wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip')
        os.system('unzip crawl-300d-2M.vec.zip')
        os.remove('crawl-300d-2M.vec.zip')

    def build_fasttext_embedding_dict(self):
        fasttext_embedding_dict = dict()
        with io.open('crawl-300d-2M.vec', 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
            headline = f.readline()
            n_fasttext, ed_fasttext = np.array(headline.split(), dtype=int)
            for line_idx in tqdm(range(n_fasttext)):
                line = f.readline()
                values = line.strip().rsplit(" ")
                word = values[0]
                if word.encode().isalpha():     # avoid iconology word like 茶道, приглашаю, مقاولات, ハイキュー, 유형,...
                    if word.islower():
                        fasttext_embedding_dict[word] = np.asarray(values[1:], dtype='float32')
                    else:
                        word = word.lower()
                        if word not in fasttext_embedding_dict:
                            fasttext_embedding_dict[word] = np.asarray(values[1:], dtype='float32')
        return fasttext_embedding_dict

    def build_embedding_matrix_FastText(self, tokenizer, embedding_dict):
        print("Building embedding matrix .....")
        if tokenizer.num_words == None:
            vocabulary_size = len(tokenizer.word_index)
        else:
            vocabulary_size = tokenizer.num_words

        zero_tokens = dict()
        embedding_matrix = np.zeros((vocabulary_size + 1, 300))

        for i in tqdm(range(1,vocabulary_size+1)):
            word = tokenizer.index_word[i]
            try:
                vec = embedding_dict[word]
                embedding_matrix[i] = vec
            except:
                zero_tokens[i] = word
        
        return embedding_matrix, zero_tokens
    
from scipy.spatial import distance
from scipy import sparse

class similarity_tool():
    def __init__(self,embedding_matrix, zero_tokens,
                 vocabulary_size = 50000, embedding_dimension=300,
                 n_jobs=4):
        self.embedding_matrix = embedding_matrix
        self.embedding_dimension = embedding_dimension
        self.zero_tokens = zero_tokens
        self.n_jobs = n_jobs
    
    def calculate_centroid_data(self, tokenized_data):
        '''-Parameter:
              tokenized_data(numpy array):
           -Return:
              (numpy array): matrix A each row is centroid vector respective tokenized_data points. 
        '''######
        # nPoints = tokenized_data.shape[0]
        # embedding_dimension = self.embedding_matrix.shape[1]
        
        with Pool(self.n_jobs) as p:
            centroid_data = p.map(self.subfunction_01_calculate_centroid_vector, tokenized_data)
        centroid_data = np.array(centroid_data)
        return centroid_data

    def subfunction_01_calculate_centroid_vector(self, tokenized_doc_i):
        tokens = tokenized_doc_i[tokenized_doc_i != 0]
        tokens = [token for token in tokens if token not in self.zero_tokens]
        if len(tokens):
            centroid_vector = np.mean(self.embedding_matrix[tokens], axis=0)
        else:
            centroid_vector = np.zeros(self.embedding_matrix.shape[1])
        return centroid_vector
    
    def calculate_matrix_cosine(self, centroid_data_01, centroid_data_02):
        '''-Parameter:
              centroid_data_01(numpy array):
              centroid_data_02(numpy array):
           -Return:
              (numpy array): matrix A(i,j) which (i,j) is cosine distance of data_(i)th and aims_(j)th 
        '''######
        self.centroid_data_aims = centroid_data_02
        self.n_aims = centroid_data_02.shape[0]
       
        with Pool(self.n_jobs) as p:
            output_matrix = p.map(self.subfunction_02_cosine, centroid_data_01)
        output_matrix = np.asarray(output_matrix)
        return output_matrix

    def subfunction_02_cosine(self,centroid_x):
        output = np.zeros(self.n_aims) 
        if np.count_nonzero(centroid_x):
            for i in range(self.n_aims):
                output[i] = distance.cosine(centroid_x, self.centroid_data_aims[i])
        return output
    
    
    
    
def save_parameter(save_object, save_file):
    with open(save_file, 'wb') as f:
        pickle.dump(save_object, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_parameter(load_file):
    with open(load_file, 'rb') as f:
        output = pickle.load(f)
    return output
