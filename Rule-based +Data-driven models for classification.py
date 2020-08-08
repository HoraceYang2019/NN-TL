# -*- coding: utf-8 -*-
'''
data: iris data for classification
model: rule-based and data-driven (GradientBoostingClassifier) models
concept: 
    1. The accociated categoreis of some samples with specific feature values can be defined as rules based on domain knowhow.
    2. When building a data-driven model from samples, 
        filter the samples defined in the rules, and use the remaining samples for modeling (data-driven).
        
    3. When testing a sample, 
        if a rule is satified then apply the domain knowledge (rule-based model), 
        otherwise apply the data-driven mode

reference:
    https://towardsdatascience.com/hybrid-rule-based-machine-learning-with-scikit-learn-9cb9841bebf2#10ad

'''
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from sklearn.base import BaseEstimator
from sklearn import metrics

# In[1]: define a class 
class RuleAugmentedGBC(BaseEstimator):
   
    def __init__(self, base_model: BaseEstimator, rules: Dict, **base_params):
        self.rules = rules
        self.base_model = base_model
        self.base_model.set_params(**base_params)
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        train_x, train_y = self._get_base_model_data(X, y)
        self.base_model.fit(train_x, train_y, **kwargs)
     
        # filter the samples defined in the rules for modeling
    def _get_base_model_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        train_x = X
      
        for feature, rul in self.rules.items():
            
            # exclude the data which are defined in the domain knowledge (rules) for modeling  
            if feature not in train_x.columns.values: continue
            for rule in rul:
              if rule[0] == "=":
                train_x = train_x.loc[train_x[feature] != rule[1]] 
              elif rule[0] == "<":
                train_x = train_x.loc[train_x[feature] >= rule[1]]
              elif rule[0] == ">":
                train_x = train_x.loc[train_x[feature] <= rule[1]]
              elif rule[0] == "<=":
                train_x = train_x.loc[train_x[feature] > rule[1]]
              elif rule[0] == ">=":
                train_x = train_x.loc[train_x[feature] < rule[1]]
              else:
                print("Invalid rule detected: {}".format(rule))
                
        indices = train_x.index.values
        train_y = y.iloc[indices] 
         # reset all indexes
        train_x = train_x.reset_index(drop=True)
        train_y = train_y.reset_index(drop=True)
        return train_x, train_y
      
    #  predict the samples by using rule-basede and data-driven models
    def predict(self, X: pd.DataFrame) -> np.array:
        p_X = X.copy()
        p_X['prediction'] = np.nan
        
        for category, rules in self.rules.items():
            if category not in p_X.columns.values: continue

          # if a rule is satified then apply the domain knowledge (rule-based model), 
          # otherwise apply the data-driven model 
            for rule in rules:
              if rule[0] == "=":
                p_X.loc[p_X[category] == rule[1], 'prediction'] = rule[2]
              elif rule[0] == "<":
                p_X.loc[p_X[category] < rule[1], 'prediction'] = rule[2]
              elif rule[0] == ">":
                p_X.loc[p_X[category] > rule[1], 'prediction'] = rule[2]
              elif rule[0] == "<=":
                p_X.loc[p_X[category] <= rule[1], 'prediction'] = rule[2]
              elif rule[0] == ">=":
                p_X.loc[p_X[category] >= rule[1], 'prediction'] = rule[2]
              else:
                print("Invalid rule detected: {}".format(rule))
                
        if len(p_X.loc[p_X['prediction'].isna()].index != 0):
            base_X = p_X.loc[p_X['prediction'].isna()].copy()
            base_X.drop('prediction', axis=1, inplace=True)
            
             # Apply the base-model for the samples un-predicted
            p_X.loc[p_X['prediction'].isna(), 'prediction'] = self.base_model.predict(base_X)
        return p_X['prediction'].values

# In[2]: access the data for classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris_data = load_iris()
data = pd.DataFrame(iris_data.data)

data.columns = iris_data['feature_names']
target = iris_data.target

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.25)
X_train.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True)
y_train = pd.Series(y_train)

# In[3]: define rules according to domain knowledge
'''for example: 
Rule 1: if sepal length < 4.3, then it should be type 1;
Rule 2: if sepal length >= 7.9, then it should be type 2. 
The rules are defined as follows.
'''
Rules = {"sepal length (cm)": [
    ("<", 4.3, 1), 
    (">=", 7.9, 2)
]}
        
# In[4]: classify the data using rule-based and data-driven models
    #
gbc = GradientBoostingClassifier(n_estimators=2)

hybrid_model = RuleAugmentedGBC(gbc, Rules)
hybrid_model.fit(X_train, y_train)
y_pred = hybrid_model.predict(X_test)
accuracy = metrics.accuracy_score(y_test, y_pred)
print(accuracy)