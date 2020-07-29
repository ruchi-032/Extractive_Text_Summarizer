#---------------------Imports-----------------

from tkinter import *
from tkinter import ttk
import pyperclip
import numpy as np
import pandas as pd
import nltk
import re
from bs4 import BeautifulSoup
import requests
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

class window:
	def __init__(self,root):
		self.root = root
		self.root.title("Text Summarizer")
		self.root.geometry("800x650+200+50")

		self.initializing()

		#--------------------Styling------------------------------------

		self.style = ttk.Style()
		self.style.map('TCombobox', fieldbackground=[('readonly','white')])
		self.style.map('TCombobox', selectbackground=[('readonly', 'white')])
		self.style.map('TCombobox', selectforeground=[('readonly', 'black')])

		#--------------------Title Space-------------------------------------

		self.title = Label(text="Text Summarizer",bd = 5, relief = GROOVE, font=("times new roman", 40, "bold"),fg = "red")
		self.title.pack(side=TOP,fill=X)

		#-------------------Main Frame--------------------------------------

		self.main_frame = Frame(self.root, bd=4, relief= GROOVE)
		self.main_frame.place(x=5,y=75,width = 793,height = 570)

		#-------------------Main frame objects------------------------------

		#-------------------Labels------------------------------------------

		self.choose_label = Label(self.main_frame, bd = 5, relief = FLAT, width=38, text="Choose an Option:",font=("times new roman",14))
		self.choose_label.grid(row=0, column = 0,padx=5,pady=5,sticky="nsew")

		#-------------------Combobox------------------------------------------

		self.combo_option = ttk.Combobox(self.main_frame,justify= 'center',state="readonly",font=("times new roman",14),width = 38)
		self.combo_option['values'] = ("Wikipedia","Text")
		self.combo_option.current(1)
		self.combo_option.grid(row=0, column = 1,padx=5,pady=5,sticky="nsew")

		#-------------------Buttons------------------------------------------

		self.sumarise_button = Button(self.main_frame,command=self.sumarise, bd = 3, relief = RAISED ,text="Summarize",height= 1,width=38)
		self.sumarise_button.grid(row=2,column=0,padx=5,pady=5,sticky="we")

		self.clear_button = Button(self.main_frame,command=self.clear, bd = 3, relief = RAISED ,text="Clear",height= 1,width=38)
		self.clear_button.grid(row=2,column=1,padx=5,pady=5,sticky="we")

		self.copy_button = Button(self.main_frame,command=self.copytext, bd = 3, relief = RAISED ,text="Copy",height= 1)
		self.copy_button.grid(row=4,column=0,columnspan= 2,padx=5,pady=5,sticky="we")

		#-------------------Textboxes------------------------------------------

		self.input_text = Text(self.main_frame, bd = 5, relief = GROOVE,height= 12,width=95)
		self.input_text.grid(row=1, column=0,columnspan=2, padx=5, pady=5,sticky='W')

		self.output_text = Text(self.main_frame, bd = 5, relief = GROOVE,height= 10,width=95)
		self.output_text.grid(row=3, column=0,columnspan=2, padx=5, pady=5, sticky='W')


	def clear(self):
		self.input_text.delete("1.0",END)
		self.output_text.delete("1.0",END)

	def initializing(self):
		#--------------------Word embedding------------------------------------

		self.word_embeddings = self.word_embed()

		#--------------------Stop words------------------------------------

		self.stop_words = stopwords.words('english')	

	def copytext(self):
		pyperclip.copy('Test')
		print("[+] Text copied")

	def wiki(self):
		self.subject = str(self.input_text.get("1.0",END))
		self.base_url = "https://en.wikipedia.org/wiki/"+self.subject.rstrip()
		self.page = requests.get(self.base_url)

		self.soup = BeautifulSoup(self.page.content,'html.parser')
		self.paragraphs = self.soup.find_all('p')

		self.content=""
		for paragraph in self.paragraphs:
			self.content+=paragraph.text
		return self.content

	def text_mode(self):
		self.content = ""
		self.content = self.input_text.get("1.0",END)
		return self.content

	def tokenize(self,text):
		self.raw_text = text
		self.sentences = sent_tokenize(self.raw_text)
		return self.sentences

	def word_embed(self):
		self.word_embedding = {}
		self.f = open('glove.6B.100d.txt', encoding='utf-8')
		for line in self.f:
			self.values = line.split()
			self.word = self.values[0]
			self.coefs = np.asarray(self.values[1:], dtype = 'float32')
			self.word_embedding[self.word] = self.coefs
		self.f.close()
		return self.word_embedding

	def clean_text(self, sentences):
		sentences = sentences
		clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")
		clean_sentences = [s.lower() for s in clean_sentences]
		return clean_sentences

	def remove_stopwords(self,sentence):
		sentences_new = " ".join([i for i in sentence if i not in self.stop_words])
		return sentences_new

	def sentence_vector(self):
		sentence_vectors = []
		for i in self.clean_sentences:
			if len(i) != 0:
				v = sum([self.word_embeddings.get(w, np.zeros((100,))) for w in i.split()])/(len(i.split())+0.001)
			else:
				v = np.zeros((100,))
			sentence_vectors.append(v)
		return sentence_vectors

	def listToString(self,sentence):  
		str2 = []
		for i in range(len(sentence)):
			str2.append(sentence[i][1])
		
		# initialize an empty string 
		str1 = " "   
		return (str1.join(list(str2)))

	def sumarise(self):
		self.output_text.delete("1.0",END)

		print("[+] Starting process")
		if self.combo_option.get() == "Wikipedia":
			self.input_user_text = self.wiki()
		else:
			self.input_user_text = self.text_mode()

		print("[+] Input text gathered")

		self.sentences = sent_tokenize(self.input_user_text)
		print("[+] Tokenization Finished")

		self.clean_sentences = self.clean_text(self.sentences)
		print("[+] Sentences cleaned")
		
		self.clean_sentences = [self.remove_stopwords(r.split()) for r in self.clean_sentences]
		print("[+] Stopwords removed")

		self.sentence_vectors = self.sentence_vector()
		print("[+] Sentence vector calculated")

		self.sim_mat = np.zeros([len(self.sentences), len(self.sentences)])
		print("[+] Array created")

		self.nx_graph = nx.from_numpy_array(self.sim_mat)
		print("[+] Graph created")

		self.scores = nx.pagerank(self.nx_graph)
		print("[+] Scores calculated")

		self.ranked_sentences = sorted(((self.scores[i], s) for i, s in enumerate(self.sentences)), reverse=True)
		print("[+] Sentences ranked")

		self.final = self.listToString(self.ranked_sentences)
		print("[+] Final text made")

		self.output_text.insert(END,self.final)
		print("[+] Output displayed")


root = Tk()
Main = window(root)
root.mainloop()



