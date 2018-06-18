import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import load_files
from PyPDF2 import PdfFileReader
from PyPDF2.pdf import PageObject
import numpy as np

# Extract the text from the .pdf file

filename = 'JavaBasics-notes.pdf'

read_obj = open(filename, 'rb+')

File = PdfFileReader(read_obj)

num_pages = File.getNumPages()
text = ''

for count in range(num_pages):
    Page = File.getPage(count)
    page_text = Page.extractText()
    text += page_text

text_folder = u'text_file'
text_subfolder = u'all'

if not os.path.exists(text_folder):
    os.makedirs(text_folder)

text_subfolder_path = os.path.join(text_folder, text_subfolder)

if not os.path.exists(text_subfolder_path):
    os.makedirs(text_subfolder_path)

write_obj_filename = os.path.join(text_subfolder_path, 'text_file.txt')
write_obj = open(write_obj_filename, 'w')
write_obj.write(text)
write_obj.close()






# Extract the features from the text

dataset = load_files(text_folder)

vectorizer = TfidfVectorizer(stop_words='english', use_idf=False)
X = vectorizer.fit_transform(dataset.data)
intermediary = np.squeeze(X[0].toarray())
Y = np.argsort(intermediary)
Z = Y[::-1][:25] # after sorting the indices by tf value, reverse the sorting and take the 25 top values
features = vectorizer.get_feature_names()
top_freq_words = [(features[i], intermediary[i]) for i in Z]
print(top_freq_words)
