# Importing the necessary third party libraries

import nltk
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

import warnings
warnings.filterwarnings('ignore')

# Importing the necessary functions from bot_functions_file module

from bot_functions_file import salutation
from bot_functions_file import advice
from bot_functions_file import LemmaWords
from bot_functions_file import FinalLemma

# If using nltk library for first time, run the below cipher to download lexical database
# nltk.download('punkt')
# nltk.download('wordnet')




# Importing the dataset with open() function in read only mode
data = open('../data/bot_data.txt', 'r')

# Converting all the words inside the dataset into lowercase
data = data.read()
data = data.lower()




# Main function for replying to client

def botReply(client_query):
    """Function for MrRobust bot replying to queries of client"""
    
    # Initialisation MrRobust reply
    MrRobust_reply = ''
    
    # Transforming the data into sentence-tokenized copy of text
    list_of_sentence = nltk.sent_tokenize(data)
    list_of_sentence.append(client_query)
    
    # Using TFidf vectorizer to transform the text to a matrix of TF-IDF attributes
    vectorizer = TfidfVectorizer(tokenizer = FinalLemma, stop_words = 'english')
    transformed_vectorizer = vectorizer.fit_transform(list_of_sentence)
    
    # Using Cosine similarity to find the similarity between queries asked by the client and the replies given by the bot
    csn_smlrty = cosine_similarity(transformed_vectorizer[-1], transformed_vectorizer)
    csn_smlrty_argsrt = csn_smlrty.argsort()[0][-2]
    csn_smlrty_flatten = csn_smlrty.flatten()
    csn_smlrty_flatten.sort()
    terminal_vectorizer = csn_smlrty_flatten[-2]
    
    # If bot is unable to understand what the client asked then it will leave a acknowledgement message
    if(terminal_vectorizer == 0):
        MrRobust_reply = MrRobust_reply + "Pardon me , I couldn't comprehend what you asked. Can you be more specific?"
        return MrRobust_reply
    
    else:
        MrRobust_reply = MrRobust_reply + list_of_sentence[csn_smlrty_argsrt]
        return MrRobust_reply[:300]




# Cipher for general queries and replies 

flag = True

# Output statement by the chatbot when the conversation commence
print("MrRobust : Hi there, I am a healthcare chatbot. You can ask me questions related to your health. " \
      "I will try my best to give an adequate reply to your queries")

while(flag == True):
    
    # For taking the queries of the client
    client_query = input()
    
    # For converting the input text into lowercase
    client_query = client_query.lower()
    
    # Replies by the bot for general queries
    if(client_query != 'exit' and client_query != 'bye'):
        if(salutation(client_query) != None):
            print("MrRobust : " + salutation(client_query) + "\n")
        elif(advice(client_query) != None):
            print("MrRobust : " + advice(client_query) + "\n")
        else:
            print("MrRobust : ", end = "")
            print(botReply(client_query) + "\n")

    # Reply by the bot at the end of conversation
    else:
        flag = False
        print("MrRobust : Bye-bye, Have a nice day ahead and take care of your health")




