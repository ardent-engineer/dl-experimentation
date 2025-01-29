#%% packages
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from scipy import spatial
import pandas as pd

#%% Data Import 
twitter_file = 'data/Tweets.csv'
df = pd.read_csv(twitter_file).dropna().sample(1000, random_state=123).reset_index(drop=True)
df.head()
# %% Sentiment Analysis
#----------------------
# sentiment_pipeline = pipeline("sentiment-analysis")
sentiment_pipeline = pipeline(model="nlptown/bert-base-multilingual-uncased-sentiment")
data = df.iloc[4]["text"]
print(sentiment_pipeline(data))

# %% Find similar Tweets
#-----------------------
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# %%
def find_closest_sentiment_text(text, df):
    emb = model.encode(data)
    df = df.assign(embeddings=df['text'].apply(lambda x: model.encode(x)))
    df['score_sim'] = df['embeddings'].apply(lambda x: spatial.distance.cosine(emb, x))
    return df.sort_values('score_sim', ascending=False).head(3)['text']

find_closest_sentiment_text(data, df)
# %%
