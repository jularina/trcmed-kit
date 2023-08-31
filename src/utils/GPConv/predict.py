import numpy as np
from src.plot.non_parametric.GPConv.plot_gp_results import plot_predictions, plot_predictions_meal, plot_predictions_conv


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
    ft_mean_meal1, ft_var_meal1 = model.predict_train(x, meals, patients_meals_idx)

    # Calculate baseline mean and covariance
    fb_mean, fb_var = model.predict_baseline(x)

    # Combine baseline and meals predictions
    f_mean, f_var = ft_mean_meal1 + fb_mean, ft_var_meal1 + fb_var

    # Extract learnt variances
    vars_learnt = [model.likelihood[i].variance.numpy().item() for i in range(P)]

    # Plot results
    metrics = plot_predictions(data, args, ids, [fb_mean, ft_mean_meal1, f_mean],
                               [fb_var, ft_var_meal1, f_var], metrics, vars_learnt=vars_learnt, time=time)

    return metrics

def predict_meal(model, args, ids, data, time, full=False):
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

    # Plot results
    ft_mean_meal1, ft_var_meal1 = model.predict_train(x, meals, patients_meals_idx)
    fb_mean, fb_var = model.predict_baseline(x)
    f_mean, f_var = ft_mean_meal1 + fb_mean, ft_var_meal1 + fb_var
    plot_predictions_meal(data, args, ids, [fb_mean, ft_mean_meal1, f_mean], [fb_var, ft_var_meal1, f_var], time=time)

def predict_conv(model, args, ids, data, time, full=True):
    x, y, meals_wo_fat, meals = data
    P = len(x)
    meals_lengths = [meal_p.shape[0] for meal_p in meals]
    patients_idx = np.arange(P, dtype=np.int32)
    patients_meals_idx = np.repeat(patients_idx, meals_lengths)

    # Calculate meals mean and covariance
    ft_mean_meal1, ft_var_meal1 = model.predict_train(x, meals, patients_meals_idx)
    ft_mean_meal1_wo, ft_var_meal1_wo = model.predict_train(x, meals_wo_fat, patients_meals_idx)

    # Calculate baseline mean and covariance
    fb_mean, fb_var = model.predict_baseline(x)

    f_mean, f_var = ft_mean_meal1 + fb_mean, ft_var_meal1 + fb_var
    f_mean_wo, f_var_wo = ft_mean_meal1_wo + fb_mean, ft_var_meal1_wo + fb_var

    # Plot results
    plot_predictions_conv(data, args, ids, [fb_mean, f_mean_wo, f_mean, ft_mean_meal1_wo, ft_mean_meal1],
                               [fb_var, f_var_wo, f_var, ft_var_meal1_wo, ft_var_meal1], time=time, full=full)
