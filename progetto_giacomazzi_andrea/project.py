import numpy as np
import pandas as pd
import re
import json
from nltk.stem import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk import word_tokenize
from nltk.corpus import stopwords
import string
import math
from collections import defaultdict
import json
import matplotlib.pyplot as plt
from operator import itemgetter
import re
from nltk.stem import PorterStemmer


def save_doc(doc, file_path): # save the documents in a json file
    with open(file_path, 'w') as file:
        json.dump(doc, file)

# function to read the documents
def read_documents():
    f = open("archive/CISI.ALL")
    merged = " "
    for line in f.readlines(): # read the lines of the file
            if line.startswith("."): # check if the line starts with a dot, which indicates the start of a new metadata
                merged += "\n" + line.strip() # add a new line to the merged variable
            else:
                merged += " " + line.strip() # add a space to the merged variable
        # updates the merged variable using a for-loop
    documents = {}
    content = ""
    doc_id = ""
    # each entry in the dictionary contains key = doc_id and value = content
    for line in merged.split("\n"):
        if line.startswith(".I"): # check if the line starts with .I, which indicates the start of a new document
            doc_id = line.split(" ")[1].strip()
        elif line.startswith(".X"): # check if the line starts with .X, which indicates the end of a document
            documents[doc_id] = content
            content = ""
            doc_id = ""
        else:
            content += line.strip()[3:] + " " # add the content of the document to the content variable
    f.close()
    return documents

def read_queries():
    f = open("archive/CISI.QRY")
    merged = ""
    for line in f.readlines():
        if line.startswith("."):
            merged += "\n" + line.strip()
        else:
            merged += " " + line.strip()
    queries = {}
    content = ""
    qry_id = ""
    for line in merged.split("\n"):
        if line.startswith(".I"):
            if not content == "":
                queries[qry_id] = content
                content = ""
                qry_id = ""
            qry_id = line.split(" ")[1].strip()
        elif line.startswith(".W") or line.startswith(".T"):
            content += line.strip()[3:] + " "
    queries[qry_id] = content
    f.close()
    return queries

def read_relevance():
    f = open("archive/CISI.REL")
    mappings = {}
    for line in f.readlines():
        voc = line.strip().split()
        key = voc[0].strip()
        current_value = voc[1].strip()
        value = []
        if key in mappings.keys():
            value = mappings.get(key)
        value.append(current_value)
        mappings[key] = value
    f.close()
    return mappings

# check if the .json files exists
# otherwise, create them
try:
    documents = json.load(open("documents.json"))
except FileNotFoundError:
    print("Creazione del file documents.json")
    documents = read_documents()
    save_doc(documents, "documents.json")

try:
    queries = json.load(open("queries.json"))
except FileNotFoundError:
    print("Creazione del file queries.json")
    queries = read_queries()
    save_doc(queries, "queries.json")

try:
    relevance = json.load(open("relevance.json"))
except FileNotFoundError:
    print("Creazione del file relevance.json")
    relevance = read_relevance()
    save_doc(relevance, "relevance.json")


## INVERTED INDEX
def normalize(text):
    no_punctuation = re.sub(r'[^\w^\s*-]','',text) # remove punctuation
    downcase = no_punctuation.lower() # lowercase
    return downcase

def tokenize(content):
    text = normalize(content)
    return list(text.split()) # return a list of tokens

def Lstemm(content):
    ps = LancasterStemmer() # stemmer
    text = tokenize(content) # tokenize
    return list(set([ps.stem(word) for word in text]))
def Pstemm(content):
    ps = PorterStemmer() # stemmer
    text = tokenize(content) # tokenize
    return list(set([ps.stem(word) for word in text])) 

# create inverted index
def create_inverted_index_P(documents):
    inverted_index = {}
    for doc_id, content in documents.items():
        #print(content)
        for token in Pstemm(content):
            if token in inverted_index.keys():
                if doc_id not in inverted_index[token]:
                    inverted_index[token].append(doc_id)
            else:
                inverted_index[token] = [doc_id]
        #if (int(doc_id) % 100 == 0):
        #    print("ID: " + str(doc_id))
    return inverted_index

try:
    inv_index_P = json.load(open("inverted_index_P.json"))
except FileNotFoundError:
    print("Creazione del file inverted_index_P.json")
    inv_index_P = create_inverted_index_P(documents)
    save_doc(inv_index_P, "inverted_index_P.json")

## BOOLEAN QUERIES
def stemm(doc):
    ps = PorterStemmer() # stemmer
    text = tokenize(doc) # tokenize
    return list(set([ps.stem(word) for word in text])) 

def build_ngram_inverted_index(documents, n): # function take in input the documents and the n-gram size
    inverted_index = {}
    print("Building ngram inverted index...")
    for doc_id, doc in enumerate(documents): # for each document
        for token in stemm(doc): # for each token in the document
            wild_token = "$" + token + "$" # add initial and final symbol
            for i in range(len(wild_token) - n + 1): # for each ngram in the token
                ngram = wild_token[i:i+n]  # extract the n-gram
                if ngram not in inverted_index:
                    inverted_index[ngram] = [] # if the ngram is not in the inverted index we add it
                if token not in inverted_index[ngram]:
                    inverted_index[ngram].append(token) # if the token is not in the inverted index we add it to the list of tokens for that ngram    
        if (doc_id % 1000 == 0):
                print("ID: " + str(doc_id))
    return inverted_index

try:
    n_gram_inv_index = json.load(open("inverted_index_ngram.json"))
except FileNotFoundError:
    print("Creazione del file inverted_index_ngram.json")
    n_gram_inv_index = build_ngram_inverted_index(documents.values(), 3)
    save_doc(n_gram_inv_index, "inverted_index_ngram.json")

# SPELLING CORRECTION using Jaccard similarity
def ngrams(word, n):
    return [word[i:i+n] for i in range(len(word)-n+1)]

def spelling_correction(word, index):    
    # Get the list of k-grams for the input word
    word_ngrams = ngrams("$" + word + "$", 3)
    # Build a set of all words that have any of these k-grams
    words_with_kgrams = set()
    for ngram in word_ngrams:
        try: # check if What are the benefits of appl* or banana?ngram is in inverted index, if not, pass
            words_with_kgrams.update(index[ngram]) 
        except KeyError:
            pass
        
    # Compute the Jaccard similarity coefficient for each candidate word,
    # and take the one that maximizes it
    scores = []
    for w in words_with_kgrams: # for each word in the set of words with k-grams
        w_ngrams = ngrams("$" + w + "$", 3) # get the list of k-grams for the word
        scores.append((w, len(set(word_ngrams).intersection(w_ngrams)) / len(set(word_ngrams).union(w_ngrams)))) # compute the Jaccard similarity coefficient and append it to the list of scores
    return max(scores, key=lambda x: x[1])[0] # return the word with the highest Jaccard similarity coefficient


# Initialize the Porter Stemmer
stemmer = PorterStemmer()

def tokenize_query(query):
    # This regex captures words, parentheses, and Boolean operators.
    tokens = re.findall(r'\(|\)|\bAND\b|\bOR\b|\bNOT\b|\w+', query, flags=re.IGNORECASE) # find all the words, parentheses, and Boolean operators
    normalized_tokens = []
    for token in tokens:
        # Check if token is boolean operator
        if token.upper() in {"AND", "OR", "NOT"}:
            normalized_tokens.append(token.upper()) # Uppercase the operator
        elif token in {"(", ")"}:
            normalized_tokens.append(token) # Keep parentheses as they are
        else:
            # For terms, lowercase and apply stemming
            token = stemmer.stem(token.lower()) # Stem the token
            if token not in inv_index_P.keys(): # Check if the token is in the index
                print(f"Token '{token}' not found in the index.")
                token = spelling_correction(token, n_gram_inv_index) # Correct the token
                print(f"Corrected to '{token}'.")
            normalized_tokens.append(token) # Append the token to the list of normalized tokens
    return normalized_tokens

def shunting_yard(tokens):  # Shunting Yard Algorithm, convert infix to postfix (reverse polish notation)
    # A AND B OR C 
    # into 
    # A B AND C OR
    
    
    output = []
    op_stack = []
    precedence = {"NOT": 3, "AND": 2, "OR": 1} # Operator precedence
    
    for token in tokens:
        if token in {"AND", "OR", "NOT"}:
            # Pop operators with higher or equal precedence
            while (op_stack and op_stack[-1] != "(" and 
                   op_stack[-1] in precedence and 
                   precedence[op_stack[-1]] >= precedence[token]):
                output.append(op_stack.pop()) # Append the operator to the output
            op_stack.append(token) # Append the operator to the stack
        elif token == "(":
            op_stack.append(token) # Append the '(' to the stack
        elif token == ")": 
            # Pop until an '(' is encountered
            while op_stack and op_stack[-1] != "(":
                output.append(op_stack.pop())
            if op_stack and op_stack[-1] == "(":
                op_stack.pop()  # Remove the '('
            else:
                raise ValueError("Mismatched parentheses in query.")
        else:
            output.append(token) # Append the token which is a term
    while op_stack:
        op = op_stack.pop()
        if op in {"(", ")"}:
            raise ValueError("Mismatched parentheses in query.")
        output.append(op)
    return output

def evaluate_postfix(postfix_tokens, inverted_index, universal_set):
    stack = []
    for token in postfix_tokens:
        if token in {"AND", "OR", "NOT"}:
            if token == "NOT":
                if not stack:
                    raise ValueError("Insufficient operands for NOT operator.")
                operand = stack.pop()
                result = universal_set - operand
                stack.append(result)
            else:
                if len(stack) < 2:
                    raise ValueError(f"Insufficient operands for {token} operator.")
                right = stack.pop()
                left = stack.pop()
                if token == "AND":
                    result = left & right  # Intersection
                elif token == "OR":
                    result = left | right  # Union
                stack.append(result)
        else:
            posting = inverted_index.get(token, set()) # Get the posting list for the term
            stack.append(posting) # Append the posting list to the stack
    
    if len(stack) != 1:
        raise ValueError(f"Error in evaluation: stack should have exactly one element, but got {stack}")
    
    return stack[0]

def evaluate_boolean_query(query, inverted_index, universal_set):
    tokens = tokenize_query(query)
    postfix = shunting_yard(tokens)
    result = evaluate_postfix(postfix, inverted_index, universal_set)
    return result 

universal_set = set(range(1, 1461))

queries2 = [
    "NOT class AND gam",
    "(NOT class) AND game",
    "class AND NOT gam",
    "class OR game",
]
converted_index = {term: set(map(int, doc_ids)) for term, doc_ids in inv_index_P.items()}

for query in queries2:
    try:
        result = evaluate_boolean_query(query, converted_index, universal_set)
        print(f"Query: {query}\nMatching Documents: {result}\n")
    except ValueError as ve:
        print(f"Query: {query}\nError: {ve}\n")

### PHRASE QUERIES

def extract_term_frequencies(text):
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words("english"))
    term_counts = defaultdict(int)

    words = word_tokenize(text.lower()) # Tokenize and convert to lowercase
    filtered_words = [stemmer.stem(word) for word in words if word not in string.punctuation and word not in stop_words]



    for word in filtered_words: 
        term_counts[word] += 1 # Count the term frequency in the document
    
    return dict(term_counts)

# Dictionary of term frequencies for each document
doc_terms = {doc_id: extract_term_frequencies(text) for doc_id, text in documents.items()}
# Dictionary of term frequencies for each query
qry_terms = {qry_id: extract_term_frequencies(text) for qry_id, text in queries.items()}

def collect_vocabulary(): # function to collect the vocabulary of the documents and queries
    all_terms = set()
    for terms in doc_terms.values():
        all_terms.update(terms.keys())

    for terms in qry_terms.values():
        all_terms.update(terms.keys())

    return sorted(all_terms) # return the sorted vocabulary

all_terms = collect_vocabulary()

def vectorize(input_features, vocabulary): # function to vectorize the documents and queries
    return {
        item_id: [input_features[item_id].get(word, 0) for word in vocabulary]
        for item_id in input_features
    }

doc_vectors = vectorize(doc_terms, all_terms)
qry_vectors = vectorize(qry_terms, all_terms)

import math
from operator import itemgetter

# Calculate vector length (Euclidean norm)
def vector_length(vector):  # norma euclidea
    return math.sqrt(sum(x ** 2 for x in vector))

# Calculate dot product of two vectors
def dot_product(vector1, vector2):
    if len(vector1) != len(vector2):
        raise ValueError("Vectors have mismatched dimensions.")
    return sum(x * y for x, y in zip(vector1, vector2) if x != 0 and y != 0)
# Compute cosine similarity
def calculate_cosine(query, document):
    return dot_product(query, document) / (vector_length(query) * vector_length(document))
# Calculate precision
def calculate_precision(model_output, gold_standard):
    return sum(1 for item in model_output if item in gold_standard) / len(model_output)
# Check if any relevant document is found
def calculate_found(model_output, gold_standard):
    return float(any(item in gold_standard for item in model_output))

# Calculate recall
def calculate_recall(model_output, gold_standard):
    return sum(1 for item in model_output if item in gold_standard) / len(gold_standard)
try: 
    precisions = np.load("precision.npy")
    recalls = np.load("recall.npy")
    total_found = np.load("total_found.npy")
except FileNotFoundError:
    # Initialize precision and found metrics
    total_precision, total_found = 0.0, 0.0
    precisions = []
    recalls = []
    print("Query ID : Precision")
    # Process all queries
    for query_id, gold_standard in relevance.items():
        query_vector = qry_vectors.get(str(query_id))
        if query_vector is None:
            continue
        
        # Compute cosine similarity for all documents
        results = {
            doc_id: calculate_cosine(query_vector, doc_vector)
            for doc_id, doc_vector in doc_vectors.items()
        }
    
        # Get top 5 most relevant documents
        model_output = [doc_id for doc_id, _ in sorted(results.items(), key=itemgetter(1), reverse=True)[:5]]
    
        # Compute precision and found metrics
        precision = calculate_precision(model_output, gold_standard)
        recall = calculate_recall(model_output, gold_standard)

        found = calculate_found(model_output, gold_standard)
    
        print(f"{query_id} : {precision:.4f}")
        #print(f"{query_id} : {found:.4f}")
    
        total_precision += precision
        total_found += found
        precisions.append(precision)
        recalls.append(recall)


    np.save("precision.npy", precisions)
    np.save("recall.npy", recalls)
    np.save("total_found.npy", total_found)

total_precision = sum(precisions)

# Compute mean precision and found ratio
num_queries = len(relevance)
print(f"MAP: {total_precision / num_queries:.4f}")
print(f"Mean Found Ratio: {total_found / num_queries:.4f}")


def plot_precision_recall_curve(i, ax):
    # Select the first query's gold standard
    query_id = list(relevance.keys())[i]  # First query
    gold_standard = relevance.get(str(query_id))  # Relevance set for the first query
    
    # Get the query vector
    query_vector = qry_vectors.get(str(query_id))
    if query_vector is None:
        print("Query vector not found.")
        exit()
    
    # Compute cosine similarity for all documents
    results = {
        doc_id: calculate_cosine(query_vector, doc_vector)
        for doc_id, doc_vector in doc_vectors.items()
    }
    
    # Sort documents by relevance based on cosine similarity
    sorted_docs = sorted(results.items(), key=itemgetter(1), reverse=True)
    
    # Lists to store precision and recall for different k values
    precision_values = []
    recall_values = []
    
    # Calculate precision and recall for different top-k values (from 1 to N)
    for k in range(1, len(sorted_docs) + 1):
        model_output = [doc_id for doc_id, _ in sorted_docs[:k]]
    
        # Calculate precision and recall for this k
        precision = calculate_precision(model_output, gold_standard)
        recall = calculate_recall(model_output, gold_standard)
    
        # Append the values to the lists
        precision_values.append(precision)
        recall_values.append(recall)
    ax.plot(recall_values, precision_values, label=f"Query {i}")
    ax.set_title(f"Query {i}")
    ax.legend()

# Create 5 subplots in a single figure
fig, axes = plt.subplots(1, 5, figsize=(15, 4))  # 1 row, 5 columns

print("Plotting precision-recall curves for the first 5 queries...")
for idx, i in enumerate(range(5)):  # Loop over 5 queries
    plot_precision_recall_curve(i, axes[idx])  # Pass the corresponding subplot

plt.tight_layout()
plt.show()

