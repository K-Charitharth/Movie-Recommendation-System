import os
import sys
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    ratings_data_path: str=os.path.join('artifacts',"ratings.csv")
    movies_data_path: str=os.path.join('artifacts',"movies.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def load_data(self):
        """
        Load ratings and movies datasets from URLs.
        
        Returns:
            tuple: Two pandas DataFrames, ratings and movies.
        """
        logging.info("Starting data ingestion process.")
        try: 
            ratings_url = 'D:/ML/projects/Movie-Recommendation-System/jupyter_notebook/Data/ratings.csv'
            movies_url = 'D:/ML/projects/Movie-Recommendation-System/jupyter_notebook/Data/movies.csv'
            
            ratings = pd.read_csv(ratings_url)
            movies = pd.read_csv(movies_url)
            # movie_ratings = ratings.merge(movies[['movieId', 'title']])

            logging.info("Data ingestion completed successfully.")
            return ratings, movies
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__ == "__main__":
    obj = DataIngestion()
    ratings, movies = obj.load_data()
    links = pd.read_csv('D:/ML/projects/Movie-Recommendation-System/jupyter_notebook/Data/links.csv')  # Contains movieId and tmdbId
    data_transformation = DataTransformation()
    X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper = data_transformation.create_utility_matrix(ratings)
    model_trainer = ModelTrainer()
    model = model_trainer.build_model(X, k = 5)
    similar_movies = model_trainer.find_similar_movies(model, 5, X, movie_mapper, movie_inv_mapper)
    print(similar_movies)
    recommendations = movies.iloc[similar_movies][['movieId', 'title']]
    recommendations['link'] = recommendations['movieId'].map(
        lambda x: f"https://www.themoviedb.org/movie/{links.loc[links['movieId'] == x, 'tmdbId'].values[0]}"
    )
    print(recommendations)

    # print(recommendations)

