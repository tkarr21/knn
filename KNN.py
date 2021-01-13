import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class KNNClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, labeltype=[], columntype=[], weight_type='inverse_distance', k=3, normalization=False):  # add parameters here
        """
        Args:
            columntype for each column tells you if continues[real] or if nominal.
            weight_type: inverse_distance voting or if non distance weighting. Options = ["no_weight","inverse_distance"]
        """
        self.k = k
        self.normalization = normalization
        self.columntype = columntype
        self.weight_type = weight_type
        self.regression = False if labeltype == 'classification' else True

    def fit(self, data, labels):
        """ Fit the data; run the algorithm (for this lab really just saves the data :D)
        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
            y (array-like): A 2D numpy array with the training targets
        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)
        """
        self.data = data
        self.labels = labels
        return self

    def predict(self, data):
        """ Predict all classes for a dataset X
        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
        Returns:
            array, shape (n_samples,)
                Predicted target values per element in X.
        """
        predicts = []
        for i in range(data.shape[0]):
            b = data[i, :]
            
            euclid = self.dist(b)
            #print(f'euclid for b^{i}: {euclid}')
            kindices = self.grab_k(euclid)
            #print(f'kindices: {kindices}')
            #print(f'k dists: {euclid[kindices]}')
            #print(f'k labels: {self.labels[kindices]}')
            
            chosen = self.voting(euclid[kindices], self.labels[kindices])
            #print(chosen)
            predicts.append(chosen)




        return predicts

    #Returns the Mean score given input data and labels
    def score(self, X, y):
        """ Return accuracy of model on a given dataset. Must implement own score function.
        Args:
                X (array-like): A 2D numpy array with data, excluding targets
                y (array-like): A 2D numpy array with targets
        Returns:
                score : float
                        Mean accuracy of self.predict(X) wrt. y.
        """
        predictions = self.predict(X)

        

        if self.regression:
            sse = 0

            for i in range(len(predictions)):
                sse += ((y[i] - predictions[i]) ** 2)

            return sse / len(predictions)

        else:
            hits = 0

            for i in range(len(predictions)):
                if y[i] == predictions[i]:
                    
                    hits += 1
            
            return hits / len(predictions)


    def dist(self, b):
        """ calculates the euclidean distance of b from each data instance
        Args:
                b (1 x n): the point to infer (must have same number of columns)
        Returns:
                euclid (1 x n): an array of b's distance from each data point 
                in self.data (provided in fit method)
        """

        diff = self.data - b
        sq_diff = np.square(diff)
        sum_sq = np.sum(sq_diff, axis=1)
        euclid = np.sqrt(sum_sq)

        return euclid

    def grab_k(self, euclid):
        """ grab the k nearest instances indice's 
        (so its easy to grab their respective labels) 
        Args:
                euclid (m x 1): 

        Returns: kindices (1 x k):
        """
        
        kindices = np.argpartition(euclid, self.k)[:self.k]
        
        return kindices


    def inv_sq_dist(self, distances):
        #element wise
        sqrd = np.square(distances)
        return np.reciprocal(sqrd, where=sqrd!=0)


    def voting(self, neighbs_dist, neighbs_labels):
        """ apply the weighting and vote for output 
        Args:
               neighbs_dist (1xk): the knn distances
               neighbs_labels (1xk): labels of the knn
        Returns:
                output of the voting
        """
        if self.regression:
            return self.regress_voting(neighbs_dist, neighbs_labels)

        
        if self.weight_type == 'inverse_distance':

            # calc inverse distances
            inv_d = self.inv_sq_dist(neighbs_dist)
            #print(f'inv_d: {inv_d}')
            return self.w_election(inv_d, neighbs_labels) 
        else:

            #grab unique values and their counts
            vals, counts = np.unique(neighbs_labels, return_counts=True)
            most_votes = np.argmax(counts)
            return vals[most_votes]


    def w_election(self, w_dist, labels):
        """ given weighted distances and respective labels 
            determine election
        """

        # grab unique values
        vals = np.unique(labels)
        
        # tuple (val, w_vote)
        best_so_far = (-1, -1)

        for val in vals:
            # calc and compare weighted votes
            val_idxs = np.where(labels == val)
            vote = np.sum(w_dist[val_idxs])
            #print(f'w_vote sum: {vote}')
            if vote > best_so_far[1]:
                best_so_far = (val, vote)

        return best_so_far[0]



    def regress_voting(self, neighbs_dist, neighbs_labels):
        
        if self.weight_type == 'inverse_distance':
           # calc inverse distances
            print(f"before weighting: {neighbs_dist}")
            inv_d = self.inv_sq_dist(neighbs_dist)
            print(f"after weighting: {inv_d}")
            return self.w_reg_eval(inv_d, neighbs_labels)

        else:
            regress_sum = np.sum(neighbs_labels)
            #print(f'labels to sum: {neighbs_labels}')
            #print(f'the sum: {regress_sum}')
            #print(f'mse: {regress_sum/ self.k}')
            return regress_sum / self.k

        
    def w_reg_eval(self, w_dist, labels):

        numerator = np.sum(labels * w_dist)
        normalize = np.sum(w_dist)
        
        print(f"labels: {labels}")
        print(f"w_dist: {w_dist}")
        print(f"numer: {numerator}")
        print(f"normer: {normalize}")
        print(f"out val: {numerator / normalize}")

        return numerator / normalize
