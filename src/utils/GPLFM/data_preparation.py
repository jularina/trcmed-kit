import numpy as np
import pandas as pd


def arrays_preparation(df):
    patients = df.loc[:, 'id'].unique()
    P = len(patients)
    x, y, meals = [], [], []

    for p in patients:
        df_patient = df[df['id'] == p]
        df_patient_glucose = df_patient[~df_patient['y'].isna()]
        df_patient_meals = df_patient[df_patient['y'].isna()]
        x_p = df_patient_glucose['t'].values.reshape(-1, 1).astype(float)
        y_p = df_patient_glucose['y'].values.reshape(-1, 1)
        meals_p = df_patient_meals[['t', 'CARBS', 'FAT']].values.reshape(-1, 3).astype(float)
        meals_p = remove_treatment_wo_effect(x_p, meals_p)
        meals_p = meals_p[meals_p[:, 1] > 5.0]
        #meals_p[:,1] = np.log(meals_p[:,1])
        #meals_p[:,2] = np.log(meals_p[:,2])

        x.append(x_p)
        y.append(y_p)
        meals.append(meals_p)

    return x, y, meals, patients, P


def remove_treatment_wo_effect(x, meals):
    meals_times = meals[:, 0]
    meals1, meals2 = meals[:, 1], meals[:, 2]
    remove_ids = []
    for i, ti in enumerate(meals_times):
        if len(x[np.logical_and(x > ti, x < ti + 1.0)]) == 0:
            remove_ids.append(i)
    meals = np.delete(meals, remove_ids, axis=0)

    return meals


def times_correction(df, df_corr, df_test=None, args=None):
    t_corr = df_corr.values.ravel('C')
    t_corr = [i for i in t_corr if i != 0]

    if args is None:
        # Only train
        j = 0
        for i in range(len(df)):
            if df.iloc[i, 6] != 0 or df.iloc[i, 7] != 0:
                df.iloc[i, 2] = t_corr[j]
                j = j + 1
        df.sort_values(by=['id', 't'], inplace=True)

        return df

    else:
        # Train and test
        # Preapre train and test corrections
        M = np.loadtxt(args.processed_data + args.period + '/M.txt', dtype='int')
        M_test = np.loadtxt(args.processed_data + args.period + '/M_test.txt', dtype='int')
        P = len(M)
        t_corr_train, t_corr_test = [], []

        for p in range(P):
            t_corr_train.extend(t_corr[:M[p]])
            t_corr_test.extend(t_corr[M[p]:M_test[p]+M[p]])
            t_corr = t_corr[M_test[p]+M[p]:]

        # Only train
        j = 0
        for i in range(len(df)):
            if df.iloc[i, 6] != 0 or df.iloc[i, 7] != 0:
                df.iloc[i, 2] = t_corr_train[j]
                j = j + 1
        df.sort_values(by=['id', 't'], inplace=True)

        # Only test
        j = 0
        for i in range(len(df_test)):
            if df_test.iloc[i, 6] != 0 or df_test.iloc[i, 7] != 0:
                df_test.iloc[i, 2] = t_corr_test[j]
                j = j + 1
        df_test.sort_values(by=['id', 't'], inplace=True)

        return df, df_test


def create_meal_prediction(m, P):
    x_agg, meals_agg = [], []
    x_p = np.linspace(0,3.0,300).reshape(-1,1)
    for p in range(P):
        carbs_av, fat_av = np.average(m[p][:,1]), np.average(m[p][:,2])
        carbs_av += 4.0
        meals_p = np.array([0.0, carbs_av, fat_av]).reshape(1,-1)
        x_agg.append(x_p)
        meals_agg.append(meals_p)

    return x_agg, meals_agg
