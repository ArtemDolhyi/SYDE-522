import numpy as np

# compute gini for a binary split
def find_gini(idx_left, idx_right, y):
    K  = np.unique(y)

    N_left = len(idx_left)
    N_right = len(idx_right)
    N = N_left + N_right

    s_left = 0
    s_right = 0
    
    for k in K:   
        p_left = sum(y[idx_left]==k) / N_left
        s_left += p_left**2

        p_right = sum(y[idx_right]==k)  / N_right
        s_right += p_right**2

    # weighted sum of split
    gini = (N_left/N)* (1-s_left) + (N_right/N)* (1-s_right)
    
    return gini


# min_leaf:
# The minimum number of samples required to be at a leaf node. 
# A split point at any depth will only be considered if it leaves
# at least min_samples_leaf training samples in each of the left and right branches.
class DecisionTree():
    def __init__(self, x, y, idxs=None, min_leaf=5):  
        if idxs is None: 
            idxs=np.arange(len(y))
        self.x, self.y, self.idxs, self.min_leaf = x, y, idxs, min_leaf   
        self.n, self.c = len(idxs), x.shape[1]
        self.val = np.mean(y[idxs])
        self.score = float('inf')
        self.check_features()
        
    @property
    def split_name(self): return self.x.columns[self.var_idx]
    
    @property
    def split_cols(self): return self.x.values[self.idxs, self.var_idx]
    
    @property
    def is_leaf(self): return self.score == float('inf')
    
    def __repr__(self):
        s = f'n: {self.n}'
        if not self.is_leaf:
            s+= f'; gini:{self.score}; split:{self.split}; var: {self.split_name}'
        return s
            
    def check_features(self):
        # loops through ALL features
        for i in range(self.c): 
            self.find_best_split(i)

        if self.is_leaf: return 
    
        #otherwise this split becomes the root of a "new tree" 
        x = self.split_cols
        lhs = np.nonzero(x <= self.split)[0]
        rhs = np.nonzero(x > self.split)[0]
        self.lhs = DecisionTree(self.x, self.y, self.idxs[lhs]) 
        self.rhs = DecisionTree(self.x, self.y, self.idxs[rhs]) 
    

    def find_best_split(self, var_idx): 
        
        x, y = self.x.values[self.idxs, var_idx], self.y[self.idxs]   

        sort_idx = np.argsort(x)
        sort_y = y[sort_idx]
        sort_x = x[sort_idx]
        
        for i in range(0, self.n-self.min_leaf-1):
            if i < self.min_leaf or sort_x[i] == sort_x[i+1]: continue 
            lhs = np.nonzero(sort_x <= sort_x[i])[0]
            rhs = np.nonzero(sort_x > sort_x[i])[0]
            if rhs.sum()==0: continue
            
            gini = find_gini(lhs, rhs, sort_y)

            if gini<self.score: 
                self.var_idx, self.score, self.split = var_idx, gini, sort_x[i]

    def predict(self, x):
        return np.array([self.predict_row(xi) for xi in x])

    def predict_row(self, xi):
        # return self.val 
        if self.is_leaf: return self.val 
        best = self.lhs if xi[self.var_idx]<=self.split else self.rhs
        return best.predict_row(xi)


# Note: current version looks at ALL features for each tree,
# instead of only checking a random subset for each split.
class RandomForest():
    def __init__ (self, x, y, n_trees, sample_sz=None, min_leaf=5):
        np.random.seed(42) 
        if sample_sz is None:
            sample_sz=len(y)

        self.x, self.y, self.sample_sz, self.min_leaf = x, y, sample_sz, min_leaf
        self.trees = [self.create_tree() for i in range(n_trees)]
    
    def create_tree(self):
        
        # bootstrapping sampling
        idxs = np.random.choice(len(self.y), replace=True, size = self.sample_sz)  

        return DecisionTree(self.x.iloc[idxs], 
                            self.y[idxs],
                            idxs=np.array(range(self.sample_sz)),
                            min_leaf=self.min_leaf)
    
    def predict(self, x):
        percents = np.mean([t.predict(x) for t in self.trees], axis=0)
        return [1 if p>0.5 else 0 for p in percents]