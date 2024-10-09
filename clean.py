import pandas as pd
from nltk.stem import WordNetLemmatizer as wnl
from nltk.corpus import stopwords
import re


stop_words = set(stopwords.words('english'))


def remove_urls(Text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')  # This regex finds URLs starting with http, https, or www
    return url_pattern.sub(r'', Text)

def remove_emojis(Text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F700-\U0001F77F"  # alchemical symbols
                           u"\U0001F780-\U0001F7FF"  # Geometric shapes extended
                           u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                           u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                           u"\U0001FA00-\U0001FA6F"  # Chess Symbols
                           u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                           u"\U00002702-\U000027B0"  # Dingbats
                           u"\U000024C2-\U0001F251"  # Enclosed characters
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', Text)

def remove_special_characters(Text):
    return re.sub(r'[^A-Za-z0-9\s]', '', Text)

def lemmatize_text(Text):
    words = Text.split()
    lemmatized_words = [wnl().lemmatize(word.lower()) for word in words if word.lower() not in stop_words]
    return ' '.join(lemmatized_words)

def clean_text(Text):
    Text = Text.lower()
    Text = remove_urls(Text)           # Remove URLs
    Text = remove_emojis(Text)         # Remove emojis
    Text = remove_special_characters(Text)  # Remove special characters
    Text = lemmatize_text(Text)
    return Text

df = pd.read_csv('sentimentdataset.csv')

df['cleaned_text'] = df['Text'].apply(clean_text)

df.to_csv('cleaned_sentiment_dataset.csv', index = False)

print("Text cleaning and conversion to lowercase completed and saved!!!")


df = pd.read_csv('cleaned_sentiment_dataset.csv')

df = df[['cleaned_text', 'Sentiment']]

print(df.head().to_string(index=False))

df.to_csv('cleansentimentdataset.csv')

print("Dataset saved!!")

#cleandataset.csv is the final dataset