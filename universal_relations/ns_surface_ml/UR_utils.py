import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
#import seaborn as sn
#from sympy import latex
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import explained_variance_score, max_error, mean_absolute_error
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import os,sys,math
from sympy import *
import time
from zipfile import ZipFile
from matplotlib import rc
rc('mathtext', fontset='cm')
import matplotlib.colors as mcolors

# Pipeline-Linear Regression model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.pipeline import Pipeline
# Grid search technique
from sklearn.model_selection import GridSearchCV
#cross validation
from sklearn.model_selection import cross_validate, LeaveOneOut

import matplotlib.cm as cm




c_color = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow',
          'brown', 'pink', 'olive', 'gray', 'lime', 'teal', 'gold', 'indigo', 'violet',
          'salmon', 'orchid', 'seagreen', 'sienna', 'darkorange', 'lightcoral', 'dodgerblue',
          'darkslategray', 'crimson', 'limegreen', 'maroon', 'peru', 'royalblue', 'slateblue',
          'mediumseagreen', 'turquoise', 'deeppink', 'darkviolet', 'darkseagreen', 'navy',
          'chocolate', 'mediumblue', 'slategray', 'cadetblue', 'forestgreen', 'orangered',
          'gold', 'cornflowerblue', 'darkcyan', 'darkolivegreen', 'rosybrown',
          'sienna', 'darkred', 'tomato', 'dimgrey', 'darkgreen', 'hotpink', 'burlywood',
          'midnightblue', 'darkblue', 'darkslateblue', 'firebrick', 'darkturquoise', 'plum',
          'greenyellow', 'black', 'lightgray', 'darkgray', 'deepskyblue',
          'lavender', 'palevioletred', 'darkmagenta', 'slategrey', 'turquoise', 'limegreen',
          'lightcoral']

##### Cross validation pipeline for polynomial model features

def cross_validation_function(data_frame, x,y,z, pol_degree):
    # Formula to predict and testing the unkown data with leaveout cross validation function
    
    #Data, targets
    train_data = data_frame[[x, y]].to_numpy()
    target = data_frame[z].to_numpy()
    
    #------Pipeline------------------------
    model = Pipeline([("poly", PolynomialFeatures(degree=pol_degree)),("linear_reg", LinearRegression())])
    
    # Scores: statistical evaluation score functions from scikit learn
    scores = ["max_error","neg_mean_absolute_error","neg_mean_squared_error","neg_root_mean_squared_error",
              "explained_variance","neg_mean_absolute_percentage_error"]
    cv_results = cross_validate(model, train_data, target, cv = LeaveOneOut(), scoring=scores, n_jobs = 3) 
    
    #------Results at cross-validation saved in Data Fame-----------
    cv_results = pd.DataFrame(cv_results)
    
    #----Statistical evaluation metric functions----------------
    #---------------calculation---------------------------------
    mean_MAE = (-cv_results['test_neg_mean_absolute_error']).mean()
    mean_MSE = (-cv_results['test_neg_mean_squared_error']).mean()
    max_validation_error = (-cv_results['test_max_error']).max()
    max_relative_valitation_error = (-100*cv_results['test_neg_mean_absolute_percentage_error']).max()
    MAPE_validation_error = (-100*cv_results['test_neg_mean_absolute_percentage_error']).mean()
    explained_variance = (cv_results['test_explained_variance']).mean()
    useful_output_data = np.array([mean_MAE,max_validation_error,mean_MSE,max_relative_valitation_error,
                          MAPE_validation_error,explained_variance,pol_degree])
    
    #--------------data saved to data frame---------------------------
    useful_data_frame = pd.DataFrame(useful_output_data)
    names = ['MAE','Max_Error','MSE','d(%)','MAPE(%)','Explained_Variance','k']
    
    #--------------Final output----------------------------------------
    validation_evaluation_metrics = pd.DataFrame(useful_data_frame.values.reshape(1,7), columns = names)
  
    
    return cv_results, validation_evaluation_metrics, useful_output_data


def cross_validation_function_2(data_frame, x,y,w,z, pol_degree):
    # Formula to predict and testing the unkown data with leaveout function
    
    #Data, targets
    train_data = data_frame[[x, y, w]].to_numpy()
    target = data_frame[z].to_numpy()
    
    #------Pipeline------------------------
    model = Pipeline([("poly", PolynomialFeatures(degree=pol_degree)),("linear_reg", LinearRegression())])
    
    # Scores: statistical evaluation score functions from scikit learn
    scores = ["max_error","neg_mean_absolute_error","neg_mean_squared_error","neg_root_mean_squared_error",
              "explained_variance","neg_mean_absolute_percentage_error"]
    cv_results = cross_validate(model, train_data, target, cv = LeaveOneOut(), scoring=scores, n_jobs = 3) #5-fold CV #LeaveOneOut()
    
    #------Results at cross-validation saved in Data Fame-----------
    cv_results = pd.DataFrame(cv_results)
    
    #----Statistical evaluation metric functions----------------
    #---------------calculation---------------------------------
    mean_MAE = (-cv_results['test_neg_mean_absolute_error']).mean()
    mean_MSE = (-cv_results['test_neg_mean_squared_error']).mean()
    max_validation_error = (-cv_results['test_max_error']).max()
    max_relative_valitation_error = (-100*cv_results['test_neg_mean_absolute_percentage_error']).max()
    MAPE_validation_error = (-100*cv_results['test_neg_mean_absolute_percentage_error']).mean()
    explained_variance = (cv_results['test_explained_variance']).mean()
    useful_output_data = np.array([mean_MAE,max_validation_error,mean_MSE,max_relative_valitation_error,
                          MAPE_validation_error,explained_variance,pol_degree])
    
    #--------------data saved to data frame---------------------------
    useful_data_frame = pd.DataFrame(useful_output_data)
    names = ['MAE','Max_Error','MSE','d(%)','MAPE(%)','Explained_Variance','k']
    
    #--------------Final output----------------------------------------
    validation_evaluation_metrics = pd.DataFrame(useful_data_frame.values.reshape(1,7), columns = names)
  
    
    return cv_results, validation_evaluation_metrics, useful_output_data



def c_val_performace_results_for_k(df,x,y,z):
    stat_metric_list_at_cv = list()
    for i in range(1,9):
        print(f'Order of the polynomial function: {i}')
        performance_results_at_cv = cross_validation_function(df, x = x,y= y ,z=z, pol_degree=i)
        print(performance_results_at_cv[1])
        stat_metric_list_at_cv.append(performance_results_at_cv[2])   
    
    names = ['MAE','Max_Error','MSE','d(%)','MAPE(%)','Explained_Variance','k']    
    df = pd.DataFrame(stat_metric_list_at_cv, columns =names)
    return df


def c_val_performace_results_for_k_2(df,x,y,w,z):
    stat_metric_list_at_cv = list()
    for i in range(1,9):
        print(f'CV for pol degree k = {i}')
        print(f'Order of the polynomial function: {i}')
        performance_results_at_cv = cross_validation_function_2(df, x = x,y= y,w=w ,z=z, pol_degree=i)
        print(performance_results_at_cv[1])
        stat_metric_list_at_cv.append(performance_results_at_cv[2])   
    
    names = ['MAE','Max_Error','MSE','d(%)','MAPE(%)','Explained_Variance','k']    
    df = pd.DataFrame(stat_metric_list_at_cv, columns =names)
    return df



####  Linear Regression function

def Regression_function(data_frame,x,y,z,pol_degree, x_power,y_power):
    # Data, targets
    train_data = data_frame[[x, y]].to_numpy()
    target = data_frame[z].to_numpy()
    
    #--------Polynomial features and defining the model---------------
    poly = PolynomialFeatures(degree=pol_degree) 
    model=LinearRegression(fit_intercept=True) #Lasso Ridge(fit_intercept=True, alpha=0.000000010)
    #-----------Applying transformation to the data-------------------
    poly_train_data = poly.fit_transform(train_data)
    
    #-----------Trainig the regression model--------------------------
    model = model.fit(poly_train_data, target)
    
    #----------Scoring: R^2 index------------------------------------
    R_square_index = model.score(poly_train_data, target)
    
    #---------z=f(x,y) prediction with respect to the model----------
    z_trial = model.predict(poly_train_data)
    z_trial = pd.DataFrame(z_trial)
    
    #----------fitting optimizers (coefficients)---------------------
    intercept_coeff = model.intercept_
    model_coeff = model.coef_
    power_combinations = poly.powers_
    powers=pd.DataFrame(poly.powers_,columns=[x_power,y_power])
   
    return z_trial, R_square_index, intercept_coeff, model_coeff, power_combinations, powers






def Regression_function_2(data_frame,x,y,w, z, pol_degree, x_power,y_power, w_power):
    # Data, targets
    train_data = data_frame[[x, y, w]].to_numpy()
    target = data_frame[z].to_numpy()
    
    #--------Polynomial features and defining the model---------------
    poly = PolynomialFeatures(degree=pol_degree) 
    model= LinearRegression(fit_intercept=True)
    #-----------Applying transformation to the data-------------------
    poly_train_data = poly.fit_transform(train_data)
    
    #-----------Trainig the regression model--------------------------
    model = model.fit(poly_train_data, target)
    
    #----------Scoring: R^2 index------------------------------------
    R_square_index = model.score(poly_train_data, target)
    
    #---------z=f(x,y) prediction with respect to the model----------
    z_trial = model.predict(poly_train_data)
    z_trial = pd.DataFrame(z_trial)
    
    #----------fitting optimizers (coefficients)---------------------
    intercept_coeff = model.intercept_
    model_coeff = model.coef_
    power_combinations = poly.powers_
    powers=pd.DataFrame(poly.powers_,columns=[x_power,y_power, w_power])
   
    return z_trial, R_square_index, intercept_coeff, model_coeff, power_combinations, powers





#### Relative errors

def relative_training_error_plot(data_frame, z , z_trial, xlabel, ylabel): 
    
    fig,ax = plt.subplots(figsize=(12, 6),)  
    labels_text_size = 20

    plt.xlabel(xlabel,size=labels_text_size)
    plt.ylabel(ylabel,size=labels_text_size)
    plt.xscale('linear')
    plt.yscale('linear')
    plt.xticks(fontsize=14, fontweight="bold")
    plt.yticks(fontsize=14, fontweight="bold")
    
    plot_representation = (100*(-data_frame[z]+data_frame[z_trial])/data_frame[z]).plot(marker='o', label = 'Relative Error'); #,figsize=(20,15)
    
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(3.0)
    
    leg = plt.legend(loc='best',framealpha = 0.8,fontsize=14)
    leg.get_frame().set_linewidth(2.0)
    leg.get_frame().set_edgecolor('black')
    
    plt.grid(False)
    plt.tight_layout()
    #save_fig = plt.savefig(fig_title,dpi=100,facecolor="w",bbox_inches='tight',transparent=True, pad_inches=0)
    plt.show()
        
    return plot_representation


def training_error_funct(data_frame, z , z_trial):
    
    # computation statistical metric functions at training data
    
    # mean relative training error
    train_mape_error = (100*abs((-data_frame[z]+data_frame[z_trial])/data_frame[z])).mean()
    
    # max training relative error calculation
    train_relat_max_errorr = np.max(100*abs(-data_frame[z]+data_frame[z_trial])/data_frame[z])
    
    # max training error calculation
    train_max_error = np.max(abs(-data_frame[z]+data_frame[z_trial]))
    
    #--------put results in an array---------------
    useful_output_data = np.array([train_max_error,train_relat_max_errorr,train_mape_error])
    
    #--------------Store data to dataframe-------------
    useful_data_frame = pd.DataFrame(useful_output_data)
    names = ['train_Max_Error','d_max_training(%)','MAPE_taining(%)']
    
    #--------------Final output------------------
    training_evaluation_metrics = pd.DataFrame(useful_data_frame.values.reshape(1,3), columns = names)
   
    return training_evaluation_metrics



def training_relative_error_hist(data_frame, z, z_trial, xlabel, ylabel,n_bins, x1_lim, x2_lim):

    fig,ax = plt.subplots(figsize=(12, 6),)
    labels_text_size = 20
    plt.xticks(fontsize=18, fontweight="bold")
    plt.yticks(fontsize=18, fontweight="bold")

    plt.xlabel(xlabel,size=labels_text_size)
    plt.ylabel(ylabel,size=labels_text_size)

    x = (100*(-data_frame[z]+data_frame[z_trial])/data_frame[z]).hist(bins=n_bins,edgecolor ='k').autoscale(enable = True, axis = 'both', tight = True)
    
    plt.xlim(x1_lim,x2_lim)
    
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(4.0)

    plt.grid(False)
    plt.tight_layout()
    plt.savefig('hist.png',dpi=100,facecolor="w",bbox_inches='tight',transparent=True, pad_inches=0.2)
    
    plt.show()
    
    
def relative_error_space_distribution(data_frame, x , y ,z, z_trial, xlabel,ylabel, zlabel, view2):
    fig = plt.figure(figsize=(10, 6)) 
    labels_text_size = 35
    
    ax = fig.add_subplot(111, projection='3d')
    
    dot_size = 40
    font_size = 30
    label_pad = 25
    label_size = 25
    
    ax.scatter3D(data_frame[x], data_frame[y], 100*abs(data_frame[z_trial]-data_frame[z])/data_frame[z], color='black', s=dot_size)    
    
    ax.view_init(30, view2)   
    ax.set_xlabel(xlabel, fontsize=font_size,labelpad=label_pad) 
    ax.set_ylabel(ylabel, fontsize=font_size,labelpad=label_pad) 
    ax.set_zlabel(zlabel, fontsize=font_size,labelpad=label_pad) 
    ax.yaxis._axinfo['label']['space_factor'] = 2.0 
    ax.xaxis._axinfo['label']['space_factor'] = 2.0 
    ax.zaxis._axinfo['label']['space_factor'] = 3.0 
    
    ax.xaxis.set_tick_params(labelsize=label_size)
    ax.yaxis.set_tick_params(labelsize=label_size)
    ax.zaxis.set_tick_params(labelsize=label_size)
    
    #for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
     #   axis.line.set_linewidth(4)
    

    ax.grid(False)
    plt.tight_layout()
    #plt.savefig('fig_title.png',dpi=100,facecolor="w",bbox_inches='tight',transparent=True, pad_inches=0.2)

    plt.show()
    
def variance_of_errors(df,reference_deviation,z,z_model):
    
    df['rel_error'] = 100*abs((z_model - z)/z)
    df_new = df[df['rel_error']>=reference_deviation] 
    return df_new


####  Surface plot $Z=F(x,y)$

def Surface_plot_funct(eos_data, x,y,z, xlabel,ylabel,zlabel, view2, n_col, border_axes):
    
    fig = plt.figure(figsize=(10, 6)) 
    labels_text_size = 15
    ax = fig.add_subplot(111, projection='3d')

    dot_size = 40
    font_size = 25
    label_pad = 24
    label_size = 15

    for i in range(0, len(eos_data)):
        ax.scatter3D(eos_data[i][x].to_numpy(),eos_data[i][y].to_numpy(),eos_data[i][z].to_numpy(),
                     s=dot_size, c=c_color[i], label = eos_labels[i])    

  
    ax.view_init(30, view2)   
    ax.set_xlabel(xlabel, fontsize=font_size,labelpad=label_pad) 
    ax.set_ylabel(ylabel, fontsize=font_size,labelpad=label_pad) 
    ax.set_zlabel(zlabel, fontsize=font_size) 
    
    ax.xaxis.set_tick_params(labelsize=label_size)
    ax.yaxis.set_tick_params(labelsize=label_size)
    ax.zaxis.set_tick_params(labelsize=label_size)
    
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.line.set_linewidth(3)
    
    ax.grid(False)

    plt.tight_layout()
    #plt.savefig('fig_title',dpi=100,facecolor="w",bbox_inches='tight',transparent=True, pad_inches=0)
    plt.show()
    
    
    
def mesh_grid_funct(number_of_points, data_frame,x_name, y_name):
    
    number_of_points = number_of_points
    
    x = np.linspace(data_frame[x_name].min(), data_frame[x_name].max(),number_of_points)
    y = np.linspace(data_frame[y_name].min(), data_frame[y_name].max(),number_of_points)
    
    X,Y = np.meshgrid(x, y)
    
    return X,Y


def mesh_grid_funct_2(number_of_points, data_frame,x_name, y_name, w_name):
    
    number_of_points = number_of_points
    
    x = np.linspace(data_frame[x_name].min(), data_frame[x_name].max(),number_of_points)
    y = np.linspace(data_frame[y_name].min(), data_frame[y_name].max(),number_of_points)
    w = np.linspace(data_frame[w_name].min(), data_frame[w_name].max(),number_of_points)
    X,Y,W = np.meshgrid(x, y, w)
    
    return X,Y,W






def Surface_plot_funct_2(eos_data, x,y,z, xlabel,ylabel,zlabel, view2, n_col, border_axes, X,Y,Z, l_w):

    #plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)
    
    fig = plt.figure(1, figsize=(12, 8)) 
    labels_text_size = 20
    ax = fig.add_subplot(projection='3d')

    dot_size = 40
    font_size = 30
    label_pad = 25
    label_size = 25

    for i in range(0, len(eos_data)):
        ax.scatter3D(eos_data[i][x].to_numpy(),eos_data[i][y].to_numpy(),eos_data[i][z].to_numpy(),
                     s=dot_size, c=c_color[i], label = eos_labels[i])    

    
    ax.plot_wireframe(X, Y, Z, rstride=40, cstride=40,edgecolor='blue' ,color = 'maroon',
                      alpha=0.6, lw = l_w, antialiased=True)

    
    
    ax.view_init(30, view2)   
    ax.set_xlabel(xlabel, fontsize=font_size, labelpad=label_pad) 
    ax.set_ylabel(ylabel, fontsize=font_size, labelpad=label_pad) 
    ax.set_zlabel(zlabel, fontsize=font_size) 
    
    
    ax.yaxis._axinfo['label']['space_factor'] = 3.0   

    ax.xaxis.set_tick_params(labelsize=label_size)
    ax.yaxis.set_tick_params(labelsize=label_size)
    ax.zaxis.set_tick_params(labelsize=label_size)
    
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.line.set_linewidth(4)
    
    ax.grid(False)

    
    plt.tight_layout()

    #plt.savefig('test.png',dpi=100,facecolor="w",bbox_inches='tight',transparent=True, pad_inches=0.2)
    plt.show()
    
    
    
    

    
    



def Surface_plot_funct_gp(eos_data, x,y,z, xlabel,ylabel,zlabel, view2, n_col, border_axes, X,Y,Z, l_w):

   # plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.5)
    
    fig = plt.figure(figsize=(12, 8)) 
    labels_text_size = 28
    ax = fig.add_subplot(111, projection='3d')

    dot_size = 30
    font_size = 30
    label_pad = 20
    label_size = 20

    for i in range(0, len(eos_data)):
        ax.scatter3D(eos_data[i][x].to_numpy(),eos_data[i][y].to_numpy(),eos_data[i][z].to_numpy(),
                     s=dot_size, c=c_color[i])  #, label = eos_labels[i]  

    
    ax.plot_wireframe(X, Y, Z, rstride=40, cstride=40,edgecolor='blue' ,color = 'maroon',
                      alpha=0.6, zorder = 15, lw = l_w, antialiased=True)
    
    
    
    ax.view_init(30, view2)   
    ax.set_xlabel(xlabel, fontsize=font_size, labelpad=label_pad) 
    ax.set_ylabel(ylabel, fontsize=font_size, labelpad=label_pad) 
    
    ax.zaxis.set_rotate_label(False) 
    ax.set_zlabel(zlabel, fontsize=font_size,labelpad=label_pad,rotation = 90)

    #ax.set_zlabel(zlabel, fontsize=font_size, labelpad=label_pad) 

    ax.zaxis.set_tick_params(pad=0)
    ax.zaxis.labelpad = 5
    

    ax.zaxis._axinfo['label']['space_factor'] = 10.0
    #ax.yaxis._axinfo['label']['space_factor'] = 1.0   
    #ax.xaxis._axinfo['label']['space_factor'] = 1.0   


    
    ax.xaxis.set_tick_params(labelsize=label_size)
    ax.yaxis.set_tick_params(labelsize=label_size)
    ax.zaxis.set_tick_params(labelsize=label_size)
    
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.line.set_linewidth(2)
    
    ax.grid(False)
    
    plt.tight_layout()

    #plt.savefig('g_p_C_sigma_2.png',dpi=100,facecolor="w",bbox_inches='tight',transparent=True, pad_inches=0.5)
    plt.show()
    
    
def Regression_function_3(data_frame,x,y,w,q, z, pol_degree, x_power,y_power, w_power, q_power):
    # Data, targets
    train_data = data_frame[[x, y, w, q]].to_numpy()
    target = data_frame[z].to_numpy()
    
    #--------Polynomial features and defining the model---------------
    poly = PolynomialFeatures(degree=pol_degree) 
    model= LinearRegression(fit_intercept=True)
    #-----------Applying transformation to the data-------------------
    poly_train_data = poly.fit_transform(train_data)
    
    #-----------Trainig the regression model--------------------------
    model = model.fit(poly_train_data, target)
    
    #----------Scoring: R^2 index------------------------------------
    R_square_index = model.score(poly_train_data, target)
    
    #---------z=f(x,y) prediction with respect to the model----------
    z_trial = model.predict(poly_train_data)
    z_trial = pd.DataFrame(z_trial)
    
    #----------fitting optimizers (coefficients)---------------------
    intercept_coeff = model.intercept_
    model_coeff = model.coef_
    power_combinations = poly.powers_
    powers=pd.DataFrame(poly.powers_,columns=[x_power,y_power, w_power, q_power])
   
    return z_trial, R_square_index, intercept_coeff, model_coeff, power_combinations, powers



    
    
    
def Surface_plot_funct_dlog(eos_data, x,y,w, z, xlabel,ylabel,wlabel,zlabel, view2, n_col,
                         border_axes, X,Y,W, Z, l_w):
    
    fig = plt.figure(figsize=(12, 8)) 
    labels_text_size = 24
    ax = fig.add_subplot(111, projection='3d')

    dot_size = 28
    font_size = 27
    label_pad = 20
    label_size = 20

    
    #######################################################################################################
   
    for i in range(0, len(eos_data)):
        scatter = ax.scatter(eos_data[i][x].to_numpy(),eos_data[i][y].to_numpy(),eos_data[i][z].to_numpy(),
                   c = eos_data[i][w].to_numpy(), s = dot_size, cmap='viridis', marker='o')    
    
    
    # Add colorbar
    cbar = plt.colorbar(scatter,  shrink=0.5)
    cbar.set_label(wlabel, fontsize=font_size, rotation=0)
    cbar.ax.tick_params(labelsize=15)
    
    
    ax.view_init(30, view2)   
    ax.set_xlabel(xlabel, fontsize=font_size,labelpad=label_pad) 
    ax.set_ylabel(ylabel, fontsize=font_size,labelpad=label_pad) 
    
    ax.zaxis.set_rotate_label(False) 
    ax.set_zlabel(zlabel, fontsize=font_size,labelpad=label_pad,rotation = 90) 
    ax.yaxis._axinfo['label']['space_factor'] = 3.0   

    ax.zaxis.set_tick_params(pad=0.)
    ax.zaxis.labelpad = 5
    
    ax.xaxis.set_tick_params(labelsize=label_size)
    ax.yaxis.set_tick_params(labelsize=label_size)
    ax.zaxis.set_tick_params(labelsize=label_size)
    
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.line.set_linewidth(2)
    
    ax.grid(False)

    
    plt.tight_layout()
    #plt.savefig('log_der_2.png',dpi=100,facecolor="w",bbox_inches='tight',transparent=True, pad_inches=0.5)
    
    plt.show()    
    
    

def EoS_categories_violin_plots(eos_categories, df, eval_metric, color_map, label, scale, y_max, y_label):
    
    combined_data = []
    max_errors = []
    for category in eos_categories:
        cat_data = np.abs(df[df['eval_eos_type'] == category][eval_metric])
        category_label = [category] * len(cat_data)
        combined_data.append((cat_data, category_label))
        max_errors.append(cat_data.max())

    # Combine the data into a Dictionary
    data_combined = {eval_metric: np.concatenate([item[0] for item in combined_data]),'EoS Category': np.concatenate([item[1] for item in combined_data])}

    combined_df = pd.DataFrame(data_combined)
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 5))    

    position_offset = 1.  # Distance multiplier to increase spacing
    positions = np.arange(len(eos_categories)) * position_offset  # Scale the positions by the multiplier

    violin_parts = ax.violinplot([np.abs(df[df['eval_eos_type'] == category][eval_metric].values) for category in eos_categories],
    positions=positions, showmeans=False, showmedians=False, 
    )

    # Assign different colors to each violin plot
    colors = cm.get_cmap(color_map, len(eos_categories))

    for i, pc in enumerate(violin_parts['bodies']):
        pc.set_facecolor(colors(i))  # Set the color for each violin plot
        pc.set_edgecolor('black')  # add edge color for contrast
        pc.set_alpha(0.8)  # Set transparency for better visibility

    # Annotate max error values
    for i, max_error in enumerate(max_errors):
        ax.annotate(
            f'{max_error:.4f}',
            (positions[i], max_error),
            textcoords="offset points",
            xytext=(0, 20),
            ha='center',
            va='center',
            color='black',
            fontsize=20,
            rotation=0
        )
    # Scatter max error points
    ax.scatter(
        positions,
        max_errors,
        color='red',
        label=label,
        zorder=5,
        marker='*',
        s=300
    )

    # Customize the plot
    ax.set_yscale(scale)
    ax.set_ylim(None, y_max)
    ax.set_xticks(positions)

    ax.set_xticklabels(eos_categories, fontsize=20)  # Remove 'pad' here
    ax.set_ylabel(y_label, fontsize=20)
    ax.tick_params(axis='y', labelsize=20)  # Adjusts font size of y-axis labels

    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(2.0)

    leg = plt.legend(loc="upper right", prop={'size': 15}, shadow=True, fontsize="large")    #,bbox_to_anchor=(1,1)
    leg.get_frame().set_linewidth(2.0)
    leg.get_frame().set_edgecolor('black')
    
    # Show plot
    plt.tight_layout()
    plt.show()
    





def EoS_class_violin_plots(eos_names, df, eval_metric, eos_class, color_map, label, scale, y_max, y_label):
    
    combined_data = []
    max_errors = []
    
    for eos_name in eos_names:
        eos_data = np.abs(df[df['eval_eos_name'] == eos_name][eval_metric])
        eos_label = [eos_name] * len(eos_data)
        combined_data.append((np.abs(eos_data), eos_label))
        max_errors.append(eos_data.max())
    
    # Combine the data into a DataFrame
    data_combined = {
        eval_metric: np.concatenate([item[0] for item in combined_data]),
        eos_class: np.concatenate([item[1] for item in combined_data])
    }
    
    combined_df = pd.DataFrame(data_combined)

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # Create violin plot with increased horizontal distance
    position_offset = 1.  # Distance multiplier to increase spacing
    positions = np.arange(len(eos_names)) * position_offset  # Scale the positions by the multiplier
    
    violin_parts = ax.violinplot(
        [np.abs(df[df['eval_eos_name'] == eos_name][eval_metric].values) for eos_name in eos_names],
        positions=positions, showmeans=False, showmedians=False, 
    )
    
    # Assign different colors to each violin plot
    colors = cm.get_cmap(color_map, len(eos_names))  # Get a color map with enough colors
    
    
    for i, pc in enumerate(violin_parts['bodies']):
        pc.set_facecolor(colors(i))  # Set the color for each violin plot
        pc.set_edgecolor('black')  # Optional: add edge color for contrast
        pc.set_alpha(0.8)  # Set transparency for better visibility
    
    # Annotate max error values
    for i, max_error in enumerate(max_errors):
        
        ax.annotate(
            f'{max_error:.4f}',
            (positions[i], max_error),
            textcoords="offset points",
            xytext=(0, 40),
            ha='center',
            va='center',
            color='black',
            fontsize=15,
            rotation=90
        )
    
    # Scatter max error points
    ax.scatter(
        positions,
        max_errors,
        color='red',
        label= label,
        zorder=5,
        marker='*',
        s=300
    )

    # Customize the plot
    ax.set_yscale('log')
    ax.set_ylim(None, y_max)
    ax.set_xticks(positions)
    
    if eos_class != 'Hybrid EoSs':
        ax.set_xticklabels(eos_names, rotation=90, fontsize=14)
    else:
        eos_names_2 = [ 'DS(CMF)-1 Hybr', 'DS(CMF)-2 Hybr', 'DS(CMF)-3 Hybr','DS(CMF)-4 Hybr', 'DS(CMF)-5 Hybr', 'DS(CMF)-6 Hybr',
       'DS(CMF)-7 Hybr', 'DS(CMF)-8 Hybr','VQCD(APR) interm', 'VQCD(APR) soft','QHC21_A', 'QHC21_AT', 'QHC21_B', 'QHC21_BT',
       'QHC21_C', 'QHC21_CT', 'QHC21_DT','DD2-vect 2 flav', 'DD2-2 flav','QHC18', 'QHC19-B', 'QHC19-C', 'QHC19-D']
        ax.set_xticklabels(eos_names_2, rotation=90, fontsize=14)

    
    ax.set_ylabel(y_label, fontsize=16)
    ax.tick_params(axis='y', labelsize=16)  # Adjusts font size of y-axis labels
    
    
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(2.0)
    
    
    leg = plt.legend(loc="upper right", prop={'size': 12}, shadow=True, fontsize="large")    #,bbox_to_anchor=(1,1)
    leg.get_frame().set_linewidth(2.0)
    leg.get_frame().set_edgecolor('black')
        
    # Show plot
    plt.tight_layout()
    
    plt.show()

































    
        
    
    
    
    
    
    
    
    
    
