import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from wordcloud import WordCloud
import nltk
from nltk.tokenize import word_tokenize,RegexpTokenizer
from nltk.stem import WordNetLemmatizer,SnowballStemmer
import re
import string
from afinn import Afinn

# ... Load your model and data ...
tfidf = pickle.load(open('tfidf.pkl','rb'))
model = pickle.load(open('Deploy.pkl','rb'))

# Write code for Cleaning Text.
# Text Cleaning Functions
lemmatizer = WordNetLemmatizer()
stemmer = SnowballStemmer('english') 
punc = string.punctuation
my_stopword = stopwords.words('english')
my_stopword.remove('not')

#Function for clean words
def preprocess(data):
    data=str(data)
    data = data.lower()

    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', data)
    
    rem_num = re.sub('[0-9]+',' ', cleantext)
    rem_num = re.sub("_+",'not', rem_num)
    rem_num = re.sub('\\w\\d\\w','',rem_num)
    rem_num = re.sub("n't",'not', rem_num)
    
    
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(rem_num)
    
    punc_word = [word for word in tokens if word not in punc]
    
    filtered_words = [word for word in punc_word if word not in my_stopword]
    
    stemming = [stemmer.stem(word) for word in filtered_words]
    
    lemma_words=[lemmatizer.lemmatize(word) for word in stemming]
    
    return " ".join(lemma_words)

# Funtion of print colorize keywords
def colorize_text(text, colors, title):
        colored_text = []
        st.subheader(title)
        for i, word in enumerate(text):
            # Use HTML to set the text color
            colored_word = f'<span style="color: {colors[i % len(colors)]};">{word}</span>'
            colored_text.append(colored_word)
            joined_text = ' '.join(colored_text)
        return st.markdown(joined_text, unsafe_allow_html=True)

# Title and Description
st.title('Hotel Review Classification')
st.write('Enter your review and we will predict the sentiment.')

# User Input
review = st.text_area('Enter your review here:', value='', height=200)

# CSS Style of Button
button_style = '''<style>
    .stButton > button {
        background-color : #330033;
        color : white;
        width: 100%;
        text-align: center;
        padding: 10px;
    }
    </style> '''

st.markdown(button_style, unsafe_allow_html=True)

Cleaning = ''
# Make a prediction
if st.button('Predict'):
    if review.strip():  # Check if the review is not empty (after stripping leading/trailing whitespace)
        Cleaning = preprocess(review)
        prediction = model.predict([Cleaning])
        if prediction == 'Positive':
            st.success('Sentiment: Positive')
        else:
            st.error('Sentiment: Negative')
    else:
        st.info('Please enter a review to predict sentiment.')

# Display Word Cloud
my_stopword = stopwords.words('english')
my_stopword.remove('not')
st.subheader('Word Cloud')

if review:
    wc = WordCloud(background_color='Black', stopwords=my_stopword).generate(str(review))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(plt)
else:
    st.warning('Please Enter a review to generate a wordcloud')

# Show Positive Negative Neutral Keywords
afinn = Afinn()
st.subheader('Review Word Sentiment')
if review.strip():
    words = Cleaning.split()
    Positive = []
    Negative = []
    Neutral = []

    for word in words:
        score = afinn.score(word)
        if score > 0:
            Positive.append(word)
        elif score <0:
            Negative.append(word)
        else:
            Neutral.append(word)

    pos_colors = ["#4CAF50", "#ADD8E6", "#9370DB"]
    colored_pos_words = colorize_text(Positive, pos_colors, 'Postive Words')

    neg_colors = ["#DF5D41 ", "#E5F073 ", "#5AF5E2"]
    colored_neg_words = colorize_text(Negative, neg_colors, 'Negative Words')

    neu_colors = ["#C2DBDB", "#5AF5E2", "#D9DDAC"]
    colored_neu_words = colorize_text(Neutral, neu_colors, 'Neutral Words')
else:
    st.warning("Please Enter a review for Words Sentiment")

# Display Data Statistics
st.subheader('Data Statistics')
st.write(f'Total Number of Words: {len(str(Cleaning).split())}')