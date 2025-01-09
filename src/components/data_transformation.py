import os
import sys
import numpy as np
from scipy.sparse import csr_matrix
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from src.utils import save_object
from sklearn.decomposition import TruncatedSVD


@dataclass
class DataTransformationConfig:
    utility_obj_file_path=os.path.join('artifacts',"utility_matrix.pkl")
    movie_mapper_obj_file = os.path.join('artifacts', 'movie_mapper.pkl')
    movie_inverse_mapper_obj_file = os.path.join('artifacts', 'movie_inverse_mapper.pkl')
    svd_reduced_obj_file_path = os.path.join('artifacts', 'svd_matrix.pkl') 


class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    # def compute_bayesian_avg(self, ratings, C, m):
    #     """
    #     Computes Bayesian Average for ratings.
        
    #     Args:
    #         ratings (pd.Series): Series of ratings.
    #         C (float): Mean count of ratings per movie.
    #         m (float): Mean of average ratings.
        
    #     Returns:
    #         float: Bayesian average rating.
    #     """
    #     return (C * m + ratings.sum()) / (C + ratings.count())

    def create_utility_matrix(self, df):
        """
        Creates a sparse utility matrix from the ratings dataframe.
        
        Args:
            df (pd.DataFrame): Ratings dataframe.

        Returns:
            tuple: Utility matrix and mapping dictionaries.
        """
        try:
            logging.info("Creating utility matrix.")
            M = df['userId'].nunique()
            N = df['movieId'].nunique()
            
            user_mapper = dict(zip(np.unique(df["userId"]), range(M)))
            movie_mapper = dict(zip(np.unique(df["movieId"]), range(N)))
            save_object(file_path=self.data_transformation_config.movie_mapper_obj_file,
                        obj = movie_mapper)
            user_inv_mapper = {v: k for k, v in user_mapper.items()}
            movie_inv_mapper = {v: k for k, v in movie_mapper.items()}
            save_object(file_path=self.data_transformation_config.movie_inverse_mapper_obj_file,
                        obj = movie_inv_mapper)
            
            user_index = df['userId'].map(user_mapper).tolist()
            item_index = df['movieId'].map(movie_mapper).tolist()
            
            X = csr_matrix((df["rating"], (user_index, item_index)), shape=(M, N))
            logging.info("Utility matrix created successfully.")
            save_object(file_path=self.data_transformation_config.utility_obj_file_path,
                        obj=X)
            return X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper
        except Exception as e:
            raise CustomException(e,sys)
        
    def svd_dimensionality_reduction(self, X, n_components=20):
        """
        Reduces dimensionality of the utility matrix using SVD.
        
        Args:
            X (csr_matrix): Utility matrix.
            n_components (int): Number of components for SVD.

        Returns:
            np.ndarray: Reduced matrix.
        """
        try:
            logging.info("Starting SVD dimensionality reduction.")
            svd = TruncatedSVD(n_components=n_components, n_iter=10)
            reduced_matrix = svd.fit_transform(X.T)
            save_object(file_path=self.data_transformation_config.svd_reduced_obj_file_path,
                        obj = reduced_matrix)
            logging.info("SVD dimensionality reduction completed.")
            return reduced_matrix
        except Exception as e:
            raise CustomException(e,sys)
