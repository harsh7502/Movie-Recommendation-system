# Movie Recommendation System

#collaborative based recommendation System



import pandas as pd
import numpy as np
from statistics import mean 

movie_data=pd.read_csv('movies.csv')
rating_data=pd.read_csv('ratings.csv')


#movie_data.iloc[8].values[1]
liked_movie_index=int(input())
#liked movie k naam input kra lo usko lower karo aur naam k lower se compare karo

#rating_data.info()

#movie_data.describe()

#movie_data=movie_data.replace(to_replace='\|',value=' ',regex=True)



movie_data=movie_data.replace({'genres':'\|'}," ",regex=True)   #iske bina bhi kaam kr rha


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
count_matrix = cv.fit_transform(movie_data.genres).toarray()

from sklearn.metrics.pairwise import cosine_similarity
similarity_scores = cosine_similarity(count_matrix)
#print(similarity_scores)

#z=rating_data['userId'][rating_data.movieId==7]

similarity_scores=pd.DataFrame(similarity_scores,index=movie_data.movieId,columns=movie_data.movieId)
#print(similarity_scores.head())
#print(similarity_scores.columns)


#df_concat = pd.concat([movie_data['movieId'], similarity_scores], axis=1)
#print(df_concat.columns)

#l=df_concat[:][df_concat.movieId==liked_movie_index]
#q=l.iloc[:,1:].values
#q=list((q))
#q=list(enumerate(q))
l=similarity_scores.loc[liked_movie_index].values  #access row by index
l=list(l)
#q=q.to_numpy().tolist()

#r=l.to_numpy().tolist()
#r=r[0]
#r=r[1:]

q=list(enumerate(l))

q.sort(key=lambda x:x[1],reverse=True)

final_movie_index=[]

for i in range(25):
    final_movie_index.append(int(movie_data.iloc[q[i][0]].values[0]))

highest_rated_index=[]
for e in final_movie_index:
    p=list(rating_data['rating'][rating_data.movieId==e].values)
    highest_rated_index.append([e,mean(p)])
    
highest_rated_index.sort(key=lambda x:x[1],reverse=True)    

recommended_movies=[]
for i in range(10):
    recommended_movies.append(movie_data['title'][movie_data.movieId==highest_rated_index[i][0]].values[0])

    

