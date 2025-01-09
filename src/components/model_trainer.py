import os
import sys
import numpy as np
from sklearn.neighbors import NearestNeighbors
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def build_model(self, X, k = 10, metric = 'cosine'):
        X = X.T
        kNN = NearestNeighbors(n_neighbors=k+1, metric=metric, algorithm="brute")
        kNN.fit(X)
        save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=kNN
            )
        return kNN




    def find_similar_movies(self, model, movie_id, X, movie_mapper, movie_inv_mapper):
        """
        Finds k-nearest neighbors for a given movie.
        
        Args:
            movie_id (int): Movie ID.
            X (csr_matrix): Utility matrix.
            movie_mapper (dict): Mapping from movie ID to index.
            movie_inv_mapper (dict): Mapping from index to movie ID.
            k (int): Number of neighbors to find.
            metric (str): Similarity metric ('cosine' or 'euclidean').

        Returns:
            list: List of similar movie IDs.
        """
        try:
            logging.info(f"Finding similar movies for movie ID {movie_id}.")
            X = X.T
            movie_ind = movie_mapper[movie_id]
            movie_vec = X[movie_ind]
            if isinstance(movie_vec, (np.ndarray)):
                movie_vec = movie_vec.reshape(1, -1)

            # kNN = NearestNeighbors(n_neighbors=k+1, metric=metric, algorithm="brute")
            # kNN.fit(X)

            neighbours = model.kneighbors(movie_vec, return_distance=False).flatten()
            similar_movies = [movie_inv_mapper[n] for n in neighbours if n != movie_ind]
            
            logging.info(f"Similar movies found for movie ID {movie_id}.")
            return similar_movies
        except Exception as e:
            raise CustomException(e,sys)

    
