#%%
import pandas as pd
from sklearn import model_selection, preprocessing
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from sklearn.metrics import mean_squared_error
#%% data import
df = pd.read_csv("ratings.csv")
df.head(2)
#%%
print(f"Unique Users: {df.userId.nunique()}, Unique Movies: {df.movieId.nunique()}")
movie_encoder = preprocessing.LabelEncoder()
user_encoder = preprocessing.LabelEncoder()

df['userId'] = user_encoder.fit_transform(df['userId'])
df['movieId'] = movie_encoder.fit_transform(df['movieId'])
#%% Data Class
class MovieDataset(Dataset):

    def __getitem__(self, idx):
        return self.users[idx], self.movies[idx], self.ratings[idx]
    def __init__(self, users, movies, ratings):
        super().__init__()
        self.users = torch.tensor(users, dtype=torch.long)
        self.ratings = torch.tensor(ratings, dtype=torch.float32)
        self.movies = torch.tensor(movies, dtype=torch.long)
        self.length = len(self.users)

    def __len__(self):
        return self.length

# %% dataset intialization    
#%% Model Class
class RecSysModel(nn.Module):
    def __init__(self, vocab_users, vocab_movies, num_e = 32):
        super().__init__()
        self.movie_embeddings = nn.Embedding(vocab_users, num_e)
        self.user_embeddings = nn.Embedding(vocab_movies, num_e)
        self.fc_1 = nn.Linear(2*num_e, 1)

    def forward(self, x_movie, x_user):
        m_emd = self.movie_embeddings(x_movie)
        u_emd = self.movie_embeddings(x_movie)
        x = torch.cat([m_emd, u_emd], dim=1)
        x = self.fc_1(x)
        return x 

#%% create train test split
df_train, df_test = model_selection.train_test_split(df, random_state=123, test_size=0.2)

#%% Dataset Instances
train_dataset = MovieDataset(
    users=df_train.userId.values,
    movies=df_train.movieId.values,
    ratings=df_train.rating.values
)

valid_dataset = MovieDataset(
    users=df_test.userId.values,
    movies=df_test.movieId.values,
    ratings=df_test.rating.values
)

#%% Data Loaders
BATCH_SIZE = 4
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True
                          ) 

test_loader = DataLoader(dataset=valid_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True
                          ) 
#%% Model Instance, Optimizer, and Loss Function
model = RecSysModel(
    vocab_users=len(user_encoder.classes_),
    vocab_movies=len(movie_encoder.classes_))

optimizer = torch.optim.Adam(model.parameters())  
criterion = nn.MSELoss()
#%% Model Training
NUM_EPOCHS = 1

model.train() 
for epoch_i in range(NUM_EPOCHS):
    for users, movies, ratings in train_loader:
        optimizer.zero_grad()
        y_pred = model(users, 
                       movies)         
        y_true = ratings.unsqueeze(dim=1).to(torch.float32)
        loss = criterion(y_pred, y_true)
        loss.backward()
        optimizer.step()

#%% Model Evaluation 
y_preds = []
y_trues = []

model.eval()
with torch.no_grad():
    for users, movies, ratings in test_loader: 
        y_true = ratings.detach().numpy().tolist()
        y_pred = model(users, movies).squeeze().detach().numpy().tolist()
        y_trues.append(y_true)
        y_preds.append(y_pred)

mse = mean_squared_error(y_trues, y_preds)
print(f"Mean Squared Error: {mse}")
#%% Users and Items
user_movie_test = {}
with torch.no_grad():
    for users, movies, ratings in test_loader:
        y_true = ratings.detach().numpy().tolist()
        y_pred = model(users, movies).squeeze().detach().numpy().tolist()
        for i in range(len(users)):
            if users[i].item() not in user_movie_test:
                user_movie_test[users[i].item()] = []
            user_movie_test[users[i].item()].append({"movie": movies[i].item(), 
                                              "y_hat": y_pred[i],
                                              "y": y_true[i]})
#%% Precision and Recall
k = 10
threshold = 3.5
user_precision = []
user_accuracy = []
for user_id, ratings in user_movie_test.items():
    ratings.sort(key=lambda x: x["y"], reverse=True)
    all_relevant_count = sum(x["y"] > threshold for x in ratings)
    predicted_as_relevant = sum(x["y_hat"] > threshold for x in ratings[:k])
    relevant_recommended = sum(x["y"] > threshold and x["y_hat"] > threshold for x in ratings[:k])
    user_precision.append(relevant_recommended/predicted_as_relevant if predicted_as_relevant !=0 else 0)
    user_accuracy.append(relevant_recommended/all_relevant_count if all_relevant_count !=0 else 0)

print(f"accuracy: {np.mean(user_accuracy)}, precision: {np.mean(user_precision)}")
# %%
