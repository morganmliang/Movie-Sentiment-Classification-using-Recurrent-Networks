# Movie-Sentiment-Classification-using-Recurrent-Networks

Code Implementation of a recurrent deep learning network to detect whether a movie review is positive or negative based on sentiment. 

Our code implements a bi-directional GRU with Self Attention Mechanism classifier that is trained on written reviews from the IMDB website. In the labeled train/dev sets, a  review that has a score <= 4 out of 10 is classified as negative, and a positive review has a score >= 7 out of 10. Thus reviews with more neutral ratings are not included.

Furthermore, the train and dev sets contain a disjoint set of movies, so no significant performance is obtained by memorizing movie-unique terms and their association with observed labels. 

Our model achieves a 89.50% classification accuracy on the full dataset.
