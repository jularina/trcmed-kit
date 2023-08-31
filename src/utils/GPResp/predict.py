import numpy as np
from src.plot.non_parametric.GPResp.plot_gp_results import plot_predictions, plot_predictions_meal, \
    plot_predictions_meal_severalsetups
import tensorflow as tf


def predict(model, args, ids, metrics, data, time):
    """ Making predictions for the whole data.

    Parameters:
    model (gpf.models.GPR): trained GPflow model
    metrics (dict): dictionary to store prediction results
    data (tuple): tuple of lists of glucose times, glucose values, meals times and values
    ids (list): list of patients' ids
    args (dict): contains input arguments
    time (str): either train or test

    Returns:
    f_mean (tf.Tensor): fitted mean glucose predictions
    """
    x, y, meals = data
    P = len(x)
    meals_lengths = [meal_p.shape[0] for meal_p in meals]
    patients_idx = np.arange(P, dtype=np.int32)
    patients_meals_idx = np.repeat(patients_idx, meals_lengths)

    # Calculate meals mean and covariance
    ft_mean_meal1, ft_mean_meal2, ft_var_meal1, ft_var_meal2 = model.predict_train(x, meals, patients_meals_idx,
                                                                                   time=time)

    # Calculate baseline mean and covariance
    fb_mean, fb_var = model.predict_baseline(x, time=time)

    # Combine baseline and meals predictions
    f_mean, f_var = ft_mean_meal1 + ft_mean_meal2 + fb_mean, ft_var_meal1 + ft_var_meal2 + fb_var

    # Extract learnt variances
    vars_learnt = [model.likelihood[i].variance.numpy().item() for i in range(P)]

    # Plot results
    metrics = plot_predictions(data, args, ids, [fb_mean, ft_mean_meal1, ft_mean_meal2, f_mean],
                               [fb_var, ft_var_meal1, ft_var_meal2, f_var], metrics, vars_learnt=vars_learnt, time=time)

    return metrics


def predict_meal(model, args, ids, metrics, data, time, order):
    """ Making predictions for the whole data.

    Parameters:
    model (gpf.models.GPR): trained GPflow model
    metrics (dict): dictionary to store prediction results
    data (tuple): tuple of lists of glucose times, glucose values, meals times and values
    ids (list): list of patients' ids
    args (dict): contains input arguments
    time (str): either train or test

    Returns:
    f_mean (tf.Tensor): fitted mean glucose predictions
    """
    x, meals = data
    P = len(x)
    meals_lengths = [meal_p.shape[0] for meal_p in meals]
    patients_idx = np.arange(P, dtype=np.int32)
    patients_meals_idx = np.repeat(patients_idx, meals_lengths)

    # Calculate meals mean and covariance
    ft_mean_meal1, ft_mean_meal2, ft_var_meal1, ft_var_meal2 = model.predict_train(x, meals, patients_meals_idx)

    # Calculate baseline mean and covariance
    fb_mean, fb_var = model.predict_baseline(x)

    # Combine baseline and meals predictions
    f_mean, f_var = ft_mean_meal1 + ft_mean_meal2 + fb_mean, ft_var_meal1 + ft_var_meal2 + fb_var

    # Plot results
    plot_predictions_meal(data, args, ids, [fb_mean, ft_mean_meal1, ft_mean_meal2, f_mean],
                          [fb_var, ft_var_meal1, ft_var_meal2, f_var], metrics, time=time, order=order)


def predict_meal_severalsetups(model, args, ids, data):
    x, meals, meals_same, meals_reverse = data
    P = len(x)
    meals_lengths = [meal_p.shape[0] for meal_p in meals]
    patients_idx = np.arange(P, dtype=np.int32)
    patients_meals_idx = np.repeat(patients_idx, meals_lengths)

    # Predictions for 1st meal setup
    ft_mean_meal1_original, ft_mean_meal2_original, ft_var_meal1_original, ft_var_meal2_original = model.predict_train(
        x, meals, patients_meals_idx)
    fb_mean, fb_var = model.predict_baseline(x)

    # Predictions for 2nd meal setup
    ft_mean_meal1_same, ft_mean_meal2_same, ft_var_meal1_same, ft_var_meal2_same = model.predict_train(x, meals_same,
                                                                                                       patients_meals_idx)

    # Predictions for 3d meal setup
    ft_mean_meal1_reverse, ft_mean_meal2_reverse, ft_var_meal1_reverse, ft_var_meal2_reverse = model.predict_train(x,
                                                                                                                   meals_reverse,
                                                                                                                   patients_meals_idx)

    # Plot results
    plot_predictions_meal_severalsetups(data, args, ids,
                                        [ft_mean_meal1_original, ft_mean_meal2_original, ft_mean_meal1_same,
                                         ft_mean_meal2_same, ft_mean_meal1_reverse, ft_mean_meal2_reverse],
                                        [ft_var_meal1_original, ft_var_meal2_original, ft_var_meal1_same,
                                         ft_var_meal2_same, ft_var_meal1_reverse, ft_var_meal2_reverse])
