"""This file contains functions that are used in main file"""

# Importing the necessary libraries

import numpy as np
import string as st
import random as rndm
import nltk
from nltk.stem import WordNetLemmatizer

import warnings
warnings.filterwarnings('ignore')


def LemmaWords(tokens):
	"""Function for lemmatization the data to create actual words instead of non-existent words"""

	lemma = WordNetLemmatizer()
	return [lemma.lemmatize(token) for token in tokens]

def FinalLemma(lists):
	"""Functions for omitting the punctuation in text and having a final lemma words"""

	omit_punctuation_dictionary = {}
	return LemmaWords(nltk.word_tokenize(lists.lower().translate(omit_punctuation_dictionary)))

# Function for general question and answering 

def salutation(questions):
    """If a user asks a question on salutation, then bot wil reply also with a randomly generated salutation"""
 
    for question in questions.split():
        if question.lower() in salutation_questions:
            return rndm.choice(salutation_reply)

def advice(aa):
    """If a user asks for health advice or tips, then bot wil reply with an randomly generated advice"""
 
    for ab in aa.split():
        if ab.lower() in advice_questions:
            return rndm.choice(advice_reply)

salutation_questions = ("hi, MrRobust", "hey", "hi", "hello", "yo", "hi, How do you do?", "Hello stranger!")
salutation_reply = ["hello", "hi, It’s nice to meet you", "hey", "hello, It’s good to see you", "hi"]

advice_questions = ("tips", "recommendation", "advice")
advice_reply = ["Eat lots of fruits and vegetables", "Consume less salt and sugar", "Eat a healthy diet",
				"Reduce intake of harmful fats", "Get active and be a healthy weight", "Do not skip breakfast"]


if __name__ == '__main__':
    main()