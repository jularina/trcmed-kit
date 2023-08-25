import numpy as np
from src.plot.plot_gp_results_combined_models import plot_predictions_meal

def predict_meal_combined_models(model, model_full, args, ids, data, time):
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

    ft_mean_meal1_full, ft_mean_meal2_full, ft_var_meal1_full, ft_var_meal2_full = model_full.predict_train(x, meals, patients_meals_idx)
    fb_mean_full, fb_var_full = model_full.predict_baseline(x)
    f_mean_full, f_var_full = ft_mean_meal1_full + ft_mean_meal2_full + fb_mean_full, ft_var_meal1_full + ft_var_meal2_full + fb_var_full

    plot_predictions_meal(data, args, ids, [fb_mean, ft_mean_meal1, f_mean],
                     [fb_var, ft_var_meal1, f_var],
                     [fb_mean_full, ft_mean_meal1_full, ft_mean_meal2_full, f_mean_full],
                     [fb_var_full, ft_var_meal1_full, ft_var_meal2_full, f_var_full], time=time)