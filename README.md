Using SVD and other methods to recommend movies
===============================================

This code presents four cases for movie recommendations:

*   Recommending movies based on ratings from similar users that only rate certain genres using cosine similarity.
*   Recommending movies based on ratings from users that rate movies from different genres (and other features) and display movies from similar users using consine similarity.
*   Recommending movies based on similarity to movies by projecting the user movies into the feature space and back. This case is covered in the thesis.
*   Predicting the user's ratings based using only SVD.

We apply the last 3 cases to real data from a movielens data set pulled from https://grouplens.org/datasets/movielens/ recommended for education and development.
The data set provided is the small size containing ~610 users, ~10K movies, ~100K ratings.
