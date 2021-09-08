import pandas as pd
import numpy as np
import statsmodels.api as sm
import math 
from sklearn.base import BaseEstimator, TransformerMixin
from lmfit import Parameters, Minimizer, report_fit, Parameters
from sklearn.metrics import r2_score, mean_squared_error, median_absolute_error

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso, Ridge, LinearRegression, ElasticNet
from sklearn.metrics import r2_score, mean_squared_error, median_absolute_error

from sklearn_pandas import DataFrameMapper, gen_features

from lmfit import Parameters, Minimizer, report_fit, Parameters

from scipy.optimize import minimize, Bounds, LinearConstraint, curve_fit

import random
import plotly.graph_objects as go



def neg_exponential_form(grp, beta, gamma, alpha=0, c=0):
    return alpha + gamma*(1 - math.exp(-beta*grp))

class DataFrameAttrsEditor(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        """
        Edits raw features
        """
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X_copy = X.copy()
        
        return X_copy
    
def MMM_dataset(dataset, dep_var, ad_vars, market_vars, extra_vars, date):
    base_feach = [x for x in ['baseline', 'seasonality', 'trend'] if x in market_vars]
    print(base_feach)
    columns=[]
    columns.append(date)
    columns=columns+dep_var+ad_vars+extra_vars+[x for x in market_vars if x not in ['baseline', 'seasonality', 'trend']]
    DT = dataset[columns]
    DT = DT.set_index(date)
    DT=DT.copy()
    
    decomposition = sm.tsa.seasonal_decompose(DT[dep_var], model='additive', period = 7)
    if 'baseline' in base_feach:
        DT['baseline']=1
    if 'seasonality' in base_feach:
        DT['seasonality']=decomposition.seasonal
    if 'trend' in base_feach:
        DT['trend']=decomposition.trend     
    return DT

class DiminishingReturnsEffectTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, gamma, beta, lambda_ad):
        """
        Diminishing Returns Effect Transformation
        
        f(x) = gamma * (1 - exp(-beta * x))
        
        Where:
            - gamma is a maximum saturation
            - beta is a saturation rate, beta > 0
        """
        self.gamma = gamma
        self.beta = beta
        self.lambda_ad = lambda_ad
        
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()
        
        X_copy = self.gamma * (1 - np.exp(-self.beta * X_copy))
        
        X_copy=AdStock(X_copy, self.lambda_ad)
        return X_copy
    
def get_data_pipeline(params,  
                      cols2impute, cols2diminish):
    
    # Imputation
    feature_imputer = [
        ([col], SimpleImputer(strategy='constant', fill_value=0))
        for col in cols2impute
    ]
    
    # Diminishing Returns Effect
    feature_diminisher = []
    for col in cols2diminish:
        col_transformer = DiminishingReturnsEffectTransformer(
            gamma=params[col + '__diminish_gamma'], 
            beta=params[col + '__diminish_beta'],
            lambda_ad=params[col + '__lambda_ad']
        )
        feature_diminisher.append((col, col_transformer))
    
    
    # Data Pipeline
    default_args = {
        'input_df': True, 
        'df_out': True, 
        'default': None
    }
    data_pipeline = Pipeline([
        ('feature_editor', DataFrameAttrsEditor()),
        ('feature_imputer', DataFrameMapper(feature_imputer, **default_args)),
        ('feature_diminisher', DataFrameMapper(feature_diminisher, **default_args))
    ])
    
    return data_pipeline

def get_prediction(params, X, cols2impute, cols2diminish):

    # Building data pipeline
    data_pipeline = get_data_pipeline(
        params=params,
        cols2impute=cols2impute, 
        cols2diminish=cols2diminish, 
    )
    
    # X_train transformation
    X_prep = data_pipeline.fit_transform(X)
    
    # Prediction
    coefs = []
    for col in X_prep.columns:
        coefs.append(params[col + '__lm_coef'])
    
    y_pred = X_prep.values.dot(coefs)
    
    return y_pred

def lmfit_objective(params, X, y, cols2impute, cols2diminish):
    
    # Building prediction
    y_pred = get_prediction(params, X, cols2impute, cols2diminish)
    
    return (y_pred - y).values

def AdStock(x, lmb=0):
    adstock = x 
    adstock[0] = (1-lmb) * x[0]

    for i in range(1, len(x)):
        adstock[i] = (1-lmb) * x[i] + lmb * adstock[i-1]  

    return adstock    
	
	




def decomposition(decomposition_df):
    colors=['aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'azure',
    'beige', 'bisque', 'black', 'blanchedalmond', 'blue',
    'blueviolet', 'brown', 'burlywood', 'cadetblue',
    'chartreuse', 'chocolate', 'coral', 'cornflowerblue',
    'cornsilk', 'crimson', 'cyan', 'darkblue', 'darkcyan',
    'darkgoldenrod', 'darkgray', 'darkgrey', 'darkgreen',
    'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange',
    'darkorchid', 'darkred', 'darksalmon', 'darkseagreen',
    'darkslateblue', 'darkslategray', 'darkslategrey',
    'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue',
    'dimgray', 'dimgrey', 'dodgerblue', 'firebrick',
    'floralwhite', 'forestgreen', 'fuchsia', 'gainsboro',
    'ghostwhite', 'gold', 'goldenrod', 'gray', 'grey', 'green',
    'greenyellow', 'honeydew', 'hotpink', 'indianred', 'indigo',
    'ivory', 'khaki', 'lavender', 'lavenderblush', 'lawngreen',
    'lemonchiffon', 'lightblue', 'lightcoral', 'lightcyan',
    'lightgoldenrodyellow', 'lightgray', 'lightgrey',
    'lightgreen', 'lightpink', 'lightsalmon', 'lightseagreen',
    'lightskyblue', 'lightslategray', 'lightslategrey',
    'lightsteelblue', 'lightyellow', 'lime', 'limegreen',
    'linen', 'magenta', 'maroon', 'mediumaquamarine',
    'mediumblue', 'mediumorchid', 'mediumpurple',
    'mediumseagreen', 'mediumslateblue', 'mediumspringgreen',
    'mediumturquoise', 'mediumvioletred', 'midnightblue',
    'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'navy',
    'oldlace', 'olive', 'olivedrab', 'orange', 'orangered',
    'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise',
    'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink',
    'plum', 'powderblue', 'purple', 'red', 'rosybrown',
    'royalblue', 'saddlebrown', 'salmon', 'sandybrown',
    'seagreen', 'seashell', 'sienna', 'silver', 'skyblue',
    'slateblue', 'slategray', 'slategrey', 'snow', 'springgreen',
    'steelblue', 'tan', 'teal', 'thistle', 'tomato', 'turquoise',
    'violet', 'wheat', 'white', 'whitesmoke', 'yellow',
    'yellowgreen']
    

def decomposition(decomposition_df):
    colors=['aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'azure',
    'beige', 'bisque', 'black', 'blanchedalmond', 'blue',
    'blueviolet', 'brown', 'burlywood', 'cadetblue',
    'chartreuse', 'chocolate', 'coral', 'cornflowerblue',
    'cornsilk', 'crimson', 'cyan', 'darkblue', 'darkcyan',
    'darkgoldenrod', 'darkgray', 'darkgrey', 'darkgreen',
    'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange',
    'darkorchid', 'darkred', 'darksalmon', 'darkseagreen',
    'darkslateblue', 'darkslategray', 'darkslategrey',
    'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue',
    'dimgray', 'dimgrey', 'dodgerblue', 'firebrick',
    'floralwhite', 'forestgreen', 'fuchsia', 'gainsboro',
    'ghostwhite', 'gold', 'goldenrod', 'gray', 'grey', 'green',
    'greenyellow', 'honeydew', 'hotpink', 'indianred', 'indigo',
    'ivory', 'khaki', 'lavender', 'lavenderblush', 'lawngreen',
    'lemonchiffon', 'lightblue', 'lightcoral', 'lightcyan',
    'lightgoldenrodyellow', 'lightgray', 'lightgrey',
    'lightgreen', 'lightpink', 'lightsalmon', 'lightseagreen',
    'lightskyblue', 'lightslategray', 'lightslategrey',
    'lightsteelblue', 'lightyellow', 'lime', 'limegreen',
    'linen', 'magenta', 'maroon', 'mediumaquamarine',
    'mediumblue', 'mediumorchid', 'mediumpurple',
    'mediumseagreen', 'mediumslateblue', 'mediumspringgreen',
    'mediumturquoise', 'mediumvioletred', 'midnightblue',
    'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'navy',
    'oldlace', 'olive', 'olivedrab', 'orange', 'orangered',
    'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise',
    'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink',
    'plum', 'powderblue', 'purple', 'red', 'rosybrown',
    'royalblue', 'saddlebrown', 'salmon', 'sandybrown',
    'seagreen', 'seashell', 'sienna', 'silver', 'skyblue',
    'slateblue', 'slategray', 'slategrey', 'snow', 'springgreen',
    'steelblue', 'tan', 'teal', 'thistle', 'tomato', 'turquoise',
    'violet', 'wheat', 'white', 'whitesmoke', 'yellow',
    'yellowgreen']


    
    fit=decomposition_df['fitted']
    act=decomposition_df['actual']
    data_help=decomposition_df.drop(columns=['fitted', 'actual'])
    col_both=[]
    col_pos=[]
    col_neg=[]
    for col in data_help:
        if col not in ['actual', 'fitted']:
            if (np.sign(data_help[col].max())-np.sign(data_help[col].min()))==2:
                col_both.append(col)
            else:
                if data_help[col].max()>0:
                    col_pos.append(col)
                else:
                    col_neg.append(col)
                    
                    
                    

    data_help=decomposition_df
    base=decomposition_df['baseline']
    col_h='baseline'

    fig = go.Figure()



    fig.add_trace(go.Scatter(x=decomposition_df.index, y=data_help[col_h].values, fill='tozeroy',
                            mode='none', name=col_h, stackgroup='two', fillcolor=random.choice(colors)  # override default markers+lines
                            ))

    for col in data_help.drop(columns=['baseline']).columns:

        if col in col_pos:
            fig.add_trace(go.Scatter(x=decomposition_df.index, y=data_help[col].values, fill='tonexty',
                            mode='none', name=col, stackgroup='two', fillcolor=random.choice(colors)   # override default markers+lines
                            ))



        if col in col_neg: 
            fig.add_trace(go.Scatter(x=decomposition_df.index, y=data_help[col].values, fill='tonexty',
                            mode='none', name=col,stackgroup='one', fillcolor=random.choice(colors)  # override default markers+lines
                            ))  


        if col in col_both:
            color=random_num = random.choice(colors)  
            fig.add_trace(go.Scatter(x=decomposition_df.index, y=data_help[col].clip(lower=0).values, fill='tonexty',
                            mode='none', name=col, stackgroup='two', fillcolor=color   # override default markers+lines
                            )) 

            fig.add_trace(go.Scatter(x=decomposition_df.index, y=data_help[col].clip(upper=0).values, fill='tonexty',
                            mode='none', name=col, stackgroup='one', fillcolor=color, showlegend=False   # override default markers+lines
                            )) 



    fig.add_trace(go.Scatter(
        x=decomposition_df.index,
        y=fit.values,
        name='Fitted', line_color="crimson"
    ))

    fig.add_trace(go.Scatter(
        x=decomposition_df.index,
        y=act.values,
        name='Actual', line_color="green"
    ))
    
    fig.update_layout(title='Decomposition of factors by its input into dependent variable',
                   xaxis_title='period',
                   yaxis_title='data')
	
	
	
    fig.show()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
    x=decomposition_df.index,
    y=decomposition_df['actual'],
    name='Actual', line_color="green"
    ))

    fig.add_trace(go.Scatter(
    x=decomposition_df.index,
    y=decomposition_df['fitted'],
    name='Fitted', line_color="crimson"
    ))
    fig.update_layout(title='Actual and fitted values of the model',
                   xaxis_title='period',
                   yaxis_title='data')
    
    fig.show()

	
	
	
def model_dt(result, X_train, y_train, cols2impute, cols2diminish):
    best_params=result.params.valuesdict()
    y_pred = get_prediction(best_params, X_train, cols2impute, cols2diminish)

    best_data_pipeline = get_data_pipeline(
        params=best_params, 
        cols2impute=cols2impute, 
        cols2diminish=cols2diminish
    )

    # X_train transformation
    X_train_prep = best_data_pipeline.fit_transform(X_train)

    best_coefs = []
    for col in X_train_prep.columns:
        best_coefs.append(best_params[col + '__lm_coef'])

    target_df = pd.DataFrame(
        data={
            'actual': y_train,
            'fitted': y_pred
        },
        index=y_train.index
    )
    decomposition_df = pd.concat((target_df, X_train_prep * best_coefs), axis=1)

    return(decomposition_df)
	
	
	
def model(params,cols2impute,cols2diminish,X_train,y_train,maxiter=200):
    data_pipeline = get_data_pipeline(
        params=params,
        cols2impute=cols2impute, 
        cols2diminish=cols2diminish
        )

    minner = Minimizer(
        userfcn=lmfit_objective,

        params=params,
        fcn_args=(X_train, y_train,cols2impute,cols2diminish)
    )


    result = minner.minimize(method='trust-constr', **{'options': {'verbose': 3, 'maxiter': maxiter}})
    return(result)
	


def ARL(x,beta,gamma,a,steep=1,EC50=.5,level=1,lag=0,adb=False,lag_na=True,window=np.nan,last=np.nan):
    arl = x
  #response curve
    if steep!=0:
        if adb==True:
            arl = AdBudg(arl,steep,EC50,level)
        else:
            arl = NegExp(arl, beta, gamma, alpha=0, c=0)
    
  #adstock  
    arl = AdStock(arl,a)

  #lags
#     if lag < 0:
#         print("????????????? ???????? lag, ???????? ???????? ???? ??????  ?? ???????!")
#     arl = shift(arl,-round(lag))

#   #NA ?????? ?? ????  
#     if lag_na==False:
#         arl[np.isnan(arl)] = 0

#     if np.isnan(last):
#         last = arl.shape[0]

#     if np.isnan(window):
#         window = last

#     arl = arl[(last-window):last]

    return arl
	
	
def shift_one(x, n):
    if n > 0:
        return np.concatenate((x[n:], np.full(n, np.nan)))[:x.shape[0]]
    elif n < 0:
        return np.concatenate((np.full(np.abs(n), np.nan), x[:n]))[:x.shape[0]]
    else:
        return x

def shift(x, shift_by):
#     return np.stack([shift_one(x, shift_by[i]) for i in range(len(shift_by))], axis=1)
    return np.stack([shift_one(x, shift_by)], axis=1)
	
	
def LogFunc(x, steep, EC50=0.5,level=1):
    cap = level*max(x) #capacity
    logfunc = x

    for i in range(len(x)):
        logfunc[i] = cap / (1+np.exp((-steep)*(x[i]/cap-EC50))) - cap/(1+np.exp(steep*EC50))

    return logfunc
	
def AdBudg(x, steep, EC50=0.5,level=1):
    cap = level #capacity
    adbudg = x

    for i in range(len(x)):
        if (x[i]==0):
            adbudg[i]=0
        else:
            adbudg[i] = cap / (1+(x[i]/(cap*EC50))**(-steep))

    return adbudg
	
def NegExp(grp, beta, gamma, alpha=0, c=0):
    return [(alpha + gamma*(1 - math.exp(-beta*x))) for x in grp]
	
	
def direct_response_curves(budg,best_params,media_costs,period=12): 
    investment=range(0,budg,10**6)
    revenue_table=pd.DataFrame()
    for chanel in media_costs.media.values:
        rev=[]
        lambda_coef=best_params[chanel+'__lambda_ad']
        beta_coef=best_params[chanel+'__diminish_beta']
        #beta_coef=0.000001
        gamma_coef=best_params[chanel+'__diminish_gamma']
        #lambda_coef=0.3
        #beta_coef=0.001
        #gamma_coef=1
        if gamma_coef>0:
            for inst in investment:
                a=inst/period/media_costs[media_costs['media']==chanel].cost.values[0]
                rev.append(sum(ARL([a for x in range(0,period)],a=lambda_coef , beta=beta_coef, gamma=gamma_coef)))
            revenue_table[chanel]=rev
        else:
            for inst in investment:
                a=inst/period/media_costs[media_costs['media']==chanel].cost.values[0]
                rev.append(-sum(ARL([a for x in range(0,period)],a=lambda_coef , beta=beta_coef, gamma=gamma_coef)))
            revenue_table[chanel]=rev
    return revenue_table
    
	
def response_curves_plot(revenue_table):
    fig = go.Figure()

    for col in revenue_table.columns:
        fig.add_trace(go.Scatter(
            x=list(range(0,revenue_table.shape[0])),
            y=revenue_table[col].values,
            name=col+'_responce'
        ))
    fig.update_layout(title='Responce curves',
                   xaxis_title='Yearly investments (mln rub)',
                   yaxis_title='Expected yearly responce (units)')
    fig.show()
    return revenue_table
	
def ROAS_curves_plot(revenue_table, budg, product_price):
    investment=range(0,budg,10**6)
    fig = go.Figure()
    revenue_table['Investment']=investment
    for col in revenue_table.drop(columns='Investment'):
        revenue_table[col+'_ROAS']=revenue_table[col]*product_price-revenue_table['Investment']
        #revenue_table[col+'_ROAS'].plot()

        fig.add_trace(go.Scatter(
            x=list(range(0,revenue_table.shape[0])),
            y=revenue_table[col+'_ROAS'].values,
            name=col+'_ROAS'
        ))
	
    fig.update_layout(title='ROAS curves',
                   xaxis_title='Yearly investments (mln rub)',
                   yaxis_title='Expected yearly ROAS (mln rub)')
	
    fig.show()
	
	
	
