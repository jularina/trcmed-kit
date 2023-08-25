import numpy as np
from src.plot.plot_gp_results_combined_periods import plot_predictions, plot_predictions_meal


def predict(model_b, model_op, args, ids, data, time):
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


    ########### OPERATION MODEL PREDICTIONS ###########
    # Calculate meals mean and covariance
    ft_mean_meal1_op, ft_mean_meal2_op, ft_var_meal1_op, ft_var_meal2_op = model_op.predict_train(x, meals, patients_meals_idx)

    # Calculate baseline mean and covariance
    fb_mean_op, fb_var_op = model_op.predict_baseline(x)

    # Combine baseline and meals predictions
    f_mean_op, f_var_op = ft_mean_meal1_op + ft_mean_meal2_op + fb_mean_op, ft_var_meal1_op + ft_var_meal2_op + fb_var_op

    ########### BASELINE MODEL PREDICTIONS ###########
    # Calculate meals mean and covariance
    ft_mean_meal1_b, ft_mean_meal2_b, ft_var_meal1_b, ft_var_meal2_b = model_b.predict_train(x, meals, patients_meals_idx)

    # Calculate baseline mean and covariance
    fb_mean_b, fb_var_b = model_b.predict_baseline(x)

    # Combine baseline and meals predictions
    f_mean_b, f_var_b = ft_mean_meal1_b + ft_mean_meal2_b + fb_mean_b, ft_var_meal1_b + ft_var_meal2_b + fb_var_b

    # Plot results
    plot_predictions(data, args, ids, [fb_mean_op, ft_mean_meal1_op, ft_mean_meal2_op, f_mean_op],
                     [fb_var_op, ft_var_meal1_op, ft_var_meal2_op, f_var_op],
                     [fb_mean_b, ft_mean_meal1_b, ft_mean_meal2_b, f_mean_b],
                     [fb_var_b, ft_var_meal1_b, ft_var_meal2_b, f_var_b], time=time)

def predict_meal(model_b, model_op, args, ids, data, time):
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


    ########### OPERATION MODEL PREDICTIONS ###########
    # Calculate meals mean and covariance
    ft_mean_meal1_op, ft_mean_meal2_op, ft_var_meal1_op, ft_var_meal2_op = model_op.predict_train(x, meals, patients_meals_idx)

    # Calculate baseline mean and covariance
    fb_mean_op, fb_var_op = model_op.predict_baseline(x)

    # Combine baseline and meals predictions
    f_mean_op, f_var_op = ft_mean_meal1_op + ft_mean_meal2_op + fb_mean_op, ft_var_meal1_op + ft_var_meal2_op + fb_var_op

    ########### BASELINE MODEL PREDICTIONS ###########
    # Calculate meals mean and covariance
    ft_mean_meal1_b, ft_mean_meal2_b, ft_var_meal1_b, ft_var_meal2_b = model_b.predict_train(x, meals, patients_meals_idx)

    # Calculate baseline mean and covariance
    fb_mean_b, fb_var_b = model_b.predict_baseline(x)

    # Combine baseline and meals predictions
    f_mean_b, f_var_b = ft_mean_meal1_b + ft_mean_meal2_b + fb_mean_b, ft_var_meal1_b + ft_var_meal2_b + fb_var_b

    # Plot results
    plot_predictions_meal(data, args, ids, [fb_mean_op, ft_mean_meal1_op, ft_mean_meal2_op, f_mean_op],
                     [fb_var_op, ft_var_meal1_op, ft_var_meal2_op, f_var_op],
                     [fb_mean_b, ft_mean_meal1_b, ft_mean_meal2_b, f_mean_b],
                     [fb_var_b, ft_var_meal1_b, ft_var_meal2_b, f_var_b], time=time)


