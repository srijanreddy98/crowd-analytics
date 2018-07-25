# # -*- coding: utf-8 -*-
# """
# Created on Sun May  6 21:20:33 2018

# @author: Rutvik
# """

# import pandas as pd
# import os
# import numpy as np

# dataset = pd.read_excel('idea_export.xlsx')

# data_use = dataset.iloc[:-10:-1,3]

# cloud_data = ""

# for i in data_use:
#     cloud_data += i

# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# import nltk
# from nltk.tokenize import RegexpTokenizer
# tokenizer = RegexpTokenizer(r'\w+')
# # nltk.download('stopwords')
# # nltk.download('punkt')

# stop_words = set(stopwords.words('english'))
# word_tokens = tokenizer.tokenize(cloud_data)


# filtered_sentence = [w for w in word_tokens if not w in stop_words]

# final_text = ""
# for i in filtered_sentence:
#     final_text += i + ' '


# import matplotlib as mpl
# import matplotlib.pyplot as plt

# from wordcloud import WordCloud

# wordcloud = WordCloud(
#                           background_color='white',
#                           max_words=200,
#                           max_font_size=40, 
#                           random_state=42
#                          ).generate(final_text)

# mpl.rcParams['font.size']=12                #10 
# mpl.rcParams['savefig.dpi']=100             #72 
# mpl.rcParams['figure.subplot.bottom']=.1 

# fig = plt.figure(1)
# plt.imshow(wordcloud)
# plt.axis('off')
# # plt.show()
# fig.savefig("word1.png", dpi=900)
print('hello world')