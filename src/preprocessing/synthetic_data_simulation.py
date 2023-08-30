import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os


def simulate_artificial_data():
    # Artificial data
    n = 3
    t0, tn = 0.0, 72.0
    patients = ['1_1', '2_1', '3_1']
    trend = [4.5, 5.6, 6.04]
    trend_p_dict = {'1_1': 4.5, '2_1': 5.6, '3_1': 6.04}
    glucose_num = [252, 234, 211]
    meals = [
        np.array([[8.0, 10, 7], [8.9, 5, 0.5], [12.5, 11, 3], [14.5, 8, 8], [20.0, 8, 5], [31.0, 6, 10],
                  [33.2, 6.5, 0.8], [37.0, 6.7, 1.0], [39.0, 8.0, 5.0], [43.5, 5.0, 5.0], [45.7, 6.5, 7.0],
                  [58.7, 10.0, 5.6], [59.5, 4.9, 1.1], [63.0, 9.0, 2.0], [65.0, 5.0, 2.5], [70.1, 6.0, 3.7]]),

        np.array([[7.5, 11.5, 10.2], [8.4, 5.0, 0.5], [13.5, 8.0, 9.3], [14.0, 5.0, 2.1], [17.3, 3.0, 0.7],
                  [18.1, 5.0, 6.5], [20.4, 8.7, 9.5],
                  [30.9, 6.9, 8.5], [37.0, 11.7, 9.3], [39.0, 4.3, 0.2], [40.2, 6.0, 5.85], [44.1, 9.7, 7.6],
                  [57.1, 13.2, 9.6], [59.5, 4.9, 1.1], [62.8, 9.7, 3.9], [64.85, 4.75, 3.5], [69.85, 8.5, 6.7]]),

        np.array([[6.6, 7.5, 8.6], [7.4, 5.0, 5.3], [13.5, 11.0, 9.3], [19.5, 5.0, 6.5], [20.5, 6.7, 6.43],
                  [32.45, 7.43, 6.5], [36.7, 6.9, 8.3], [39.0, 7.3, 0.8], [41.2, 5.72, 5.85], [45.1, 7.7, 4.6],
                  [56.5, 13.2, 9.6], [60.3, 4.8, 3.1], [63.1, 9.5, 7.47], [69.35, 4.75, 5.06]])

    ]

    b_carbs, b_fat = [0.08, 0.072, 0.06], [0.04, 0.02, 0.03]
    l = [0.4, 0.35, 0.5]

    # Save to train/test dataframe
    df = pd.DataFrame(columns=['id', 't', 'y', 'CARBS', 'FAT'])

    # Simulate artificial glucose values
    for p in range(n):
        # Save meals data to dataframe
        for m in meals[p]:
            new_row = {'id': patients[p], 't': m[0], 'y': np.nan, 'CARBS': m[1], 'FAT': m[2]}
            df.loc[len(df)] = new_row

        # Save glucose data to dataframe
        glucose_times = np.linspace(t0, tn, glucose_num[p])
        noise = np.random.normal(0, 1, glucose_num[p])
        glucose_times = glucose_times + noise
        glucose_times = np.sort(glucose_times)

        glucose = []
        for t in glucose_times:
            sum_meals = 0
            for m in meals[p]:
                if t > m[0]:
                    sum_meals += (b_carbs[p] * m[1] + b_fat[p] * m[2]) * np.exp(
                        (-0.5) * ((t - m[0] - 3 * l[p])) ** 2 / (l[p] ** 2))
                    if t <= 3:
                        sum_meals += np.random.normal(0, 0.5, 1)[0]
            sum_meals += trend[p]
            glucose.append(sum_meals)

            new_row = {'id': patients[p], 't': t, 'y': sum_meals, 'CARBS': 0.0, 'FAT': 0.0}
            df.loc[len(df)] = new_row

        # plt.plot(glucose_times, glucose, c="midnightblue", linewidth=2, label='True glucose curve', zorder=2)
        # plt.plot(glucose_times, glucose, 'kx', ms=7, alpha=0.7, label='True observations')
        # plt.show()

    df.sort_values(by=['id', 't'], inplace=True)
    df_train = df[df['t'] <= 48]
    df_test = df[df['t'] > 48]

    return df_train, df_test


def train_data_save(df_sliced, path):
    trend_p_dict = {'1_1': 4.5, '2_1': 5.6, '3_1': 6.04}
    P = len(df_sliced['id'].unique())
    patients = df_sliced['id'].unique()
    N, M, trend_p = np.array([], dtype=np.int8), np.array([], dtype=np.int8), np.array([], dtype=np.int8)

    for p in patients:
        df_patient = df_sliced[df_sliced['id'] == p]
        df_patient1 = df_patient[~df_patient['y'].isna()]
        df_patient2 = df_patient[df_patient['y'].isna()]

        n = df_patient1.shape[0]  # Number of glucose observations for each patient
        N = np.append(N, n)

        m = df_patient2.shape[0]  # Number of meals for each patient
        M = np.append(M, m)

        trend_p = np.append(trend_p, trend_p_dict[p])

    N_max = max(N)
    M_max = max(M)
    T_max = df_sliced['t'].max()

    y = np.zeros((P, N_max))
    t = np.zeros((P, N_max))
    x1 = np.array([])  # Storage for carbs
    x2 = np.array([])  # Storage for fat
    x = np.array([])
    tx = np.array([])

    for p in range(P):
        df_patient = df_sliced[df_sliced['id'] == patients[p]]
        df_patient1 = df_patient[~df_patient['y'].isna()]
        df_patient2 = df_patient[df_patient['y'].isna()]

        df_patient1_glucose = df_patient1['y']  # Glucose observations for each patient
        df_patient1_glucose_time = df_patient1['t']  # Glucose time observations for each patient
        y[p, :N[p]] = df_patient1_glucose
        t[p, :N[p]] = df_patient1_glucose_time

        df_patient2_meals_carbs = df_patient2['CARBS']  # Carbs meals observations for each patient
        df_patient2_meals_fat = df_patient2['FAT']  # Fat meals observations for each patient
        df_patient2_meals = df_patient2['CARBS'] + df_patient2['FAT']  # Meals observations for each patient

        df_patient2_meals_time = df_patient2['t']  # Meals time observations for each patient
        x1 = np.append(x1, df_patient2_meals_carbs)
        x2 = np.append(x2, df_patient2_meals_fat)
        x = np.append(x, df_patient2_meals)
        tx = np.append(tx, df_patient2_meals_time)

    PM = len(x1)

    # Open files for saving processed data
    os.makedirs(path, exist_ok=True)
    file_params = open(path + "params.txt", "w")
    file_params.write('\n'.join([str(P), str(N_max), str(M_max), str(T_max), str(PM)]))
    file_params.close()

    np.savetxt(path + "N.txt", N, fmt=['%d'])
    np.savetxt(path + "M.txt", M, fmt=['%d'])
    np.savetxt(path + "trend_p.txt", trend_p)
    np.savetxt(path + "y.txt", y)
    np.savetxt(path + "t.txt", t)
    np.savetxt(path + "x1.txt", x1)
    np.savetxt(path + "x2.txt", x2)
    np.savetxt(path + "x.txt", x)
    np.savetxt(path + "tx.txt", tx)
    np.savetxt(path + "patients.txt", patients, fmt=['%s'])

    df_sliced.to_csv(path + "df_sliced.csv")

def test_data_save(df_sliced_test, path):
    """Create and save training data

    Parameters:
    path (str): path to training data folder
    df_sliced_test (pd.Dataframe): cut test dataframe of patients' meals/glucose data

    Returns:
    -
   """
    P = len(df_sliced_test['id'].unique())
    patients = df_sliced_test['id'].unique()
    N_test, M_test = np.array([], dtype=np.int8), np.array([], dtype=np.int8)

    for p in patients:
        df_patient = df_sliced_test[df_sliced_test['id'] == p]
        df_patient1 = df_patient[~df_patient['y'].isna()]
        df_patient2 = df_patient[df_patient['y'].isna()]

        n = df_patient1.shape[0]  # Number of glucose observations for each patient
        N_test = np.append(N_test, n)

        m = df_patient2.shape[0]  # Number of meals for each patient
        M_test = np.append(M_test, m)

    N_max_test = max(N_test)
    M_max_test = max(M_test)
    T_max_test = df_sliced_test['t'].max()

    y_test = np.zeros((P, N_max_test))
    t_test = np.zeros((P, N_max_test))
    x1_test = np.array([])  # Storage for carbs
    x2_test = np.array([])  # Storage for fat
    x_test = np.array([])
    tx_test = np.array([])

    for p in range(P):
        df_patient = df_sliced_test[df_sliced_test['id'] == patients[p]]
        df_patient1 = df_patient[~df_patient['y'].isna()]
        df_patient2 = df_patient[df_patient['y'].isna()]

        df_patient1_glucose = df_patient1['y']  # Glucose observations for each patient
        df_patient1_glucose_time = df_patient1['t']  # Glucose time observations for each patient
        y_test[p, :N_test[p]] = df_patient1_glucose
        t_test[p, :N_test[p]] = df_patient1_glucose_time

        df_patient2_meals_carbs = df_patient2['CARBS']# Carbs meals observations for each patient
        df_patient2_meals_fat = df_patient2['FAT']  # Fat meals observations for each patient
        df_patient2_meals = df_patient2['CARBS'] + df_patient2['FAT']  # Meals observations for each patient

        df_patient2_meals_time = df_patient2['t']  # Meals time observations for each patient
        x1_test = np.append(x1_test, df_patient2_meals_carbs)
        x2_test = np.append(x2_test, df_patient2_meals_fat)
        x_test = np.append(x_test, df_patient2_meals)
        tx_test = np.append(tx_test, df_patient2_meals_time)

    PM_test = len(x1_test)

    # Open files for saving processed data
    os.makedirs(path, exist_ok=True)
    file_params = open(path + "params_test.txt", "w")
    file_params.write('\n'.join([str(P), str(N_max_test), str(M_max_test), str(T_max_test), str(PM_test)]))
    file_params.close()

    np.savetxt(path + "N_test.txt", N_test, fmt=['%d'])
    np.savetxt(path + "M_test.txt", M_test, fmt=['%d'])
    np.savetxt(path + "y_test.txt", y_test)
    np.savetxt(path + "t_test.txt", t_test)
    np.savetxt(path + "x1_test.txt", x1_test)
    np.savetxt(path + "x2_test.txt", x2_test)
    np.savetxt(path + "x_test.txt", x_test)
    np.savetxt(path + "tx_test.txt", tx_test)

    df_sliced_test.to_csv(path + "df_sliced_test.csv")

def concat_train_test(x, x_test, M, M_test):
    x_combo = []
    P = len(M)
    for p in range(P):
        x_combo.extend(x[:M[p]])
        x_combo.extend(x_test[:M_test[p]])
        x = x[M[p]:]
        x_test = x_test[M_test[p]:]

    return x_combo

def combine_train_test(path):
    N = np.loadtxt(path + "N.txt", dtype='int')
    N_test = np.loadtxt(path + "N_test.txt", dtype='int')
    N_combo = N+N_test
    np.savetxt(path + "N_combo.txt", N_combo, fmt=['%d'])

    M = np.loadtxt(path + "M.txt", dtype='int')
    M_test = np.loadtxt(path + "M_test.txt", dtype='int')
    M_combo = M+M_test
    np.savetxt(path + "M_combo.txt", M_combo, fmt=['%d'])

    P, N_max, M_max = len(N_combo), int(max(N_combo)), int(max(M_combo))

    y = np.loadtxt(path + "y.txt")
    y_test = np.loadtxt(path + "y_test.txt")
    t = np.loadtxt(path + "t.txt")
    t_test = np.loadtxt(path + "t_test.txt")
    y_combo = np.zeros((P, N_max))
    t_combo = np.zeros((P, N_max))

    for p in range(P):
        y_combo[p, :N_combo[p]] = np.concatenate((y[p, :N[p]], y_test[p, :N_test[p]]))
        t_combo[p, :N_combo[p]] = np.concatenate((t[p, :N[p]], t_test[p, :N_test[p]]))

    np.savetxt(path + "y_combo.txt", y_combo)
    np.savetxt(path + "t_combo.txt", t_combo)

    x1 = np.loadtxt(path + "x1.txt")
    x1_test = np.loadtxt(path + "x1_test.txt")
    x1_combo = concat_train_test(x1, x1_test, M, M_test)
    np.savetxt(path + "x1_combo.txt", x1_combo)

    x2 = np.loadtxt(path + "x2.txt")
    x2_test = np.loadtxt(path + "x2_test.txt")
    x2_combo = concat_train_test(x2, x2_test, M, M_test)
    np.savetxt(path + "x2_combo.txt", x2_combo)

    x = np.loadtxt(path + "x.txt")
    x_test = np.loadtxt(path + "x_test.txt")
    x_combo = concat_train_test(x, x_test, M, M_test)
    np.savetxt(path + "x_combo.txt", x_combo)

    tx = np.loadtxt(path + "tx.txt")
    tx_test = np.loadtxt(path + "tx_test.txt")
    tx_combo = concat_train_test(tx, tx_test, M, M_test)
    np.savetxt(path + "tx_combo.txt", tx_combo)

    params_combo = np.array([P, N_max, M_max, t_combo.max(), len(x1_combo)])
    np.savetxt(path + "params_combo.txt", params_combo, fmt=['%d'])


if __name__ == "__main__":
    os.chdir("../../")
    path = "./data/synthetic/processed_data/"
    df_train, df_test = simulate_artificial_data()
    train_data_save(df_train, path)
    test_data_save(df_test, path)
    combine_train_test(path)
