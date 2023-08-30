import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import msoffcrypto
import io
from glob import glob
import argparse

# !pip install msoffcrypto-tool
pd.options.mode.chained_assignment = None

parser = argparse.ArgumentParser('Preprocessing initial data for stan model.')
parser.add_argument('--period', type=str, default='operation',
                    help="Time period to be considered.")
parser.add_argument('--patient_data', type=str,
                    default='../../data/raw_data/data_files/Libredata_16042020_password_protected'
                            '.xlsx',
                    help="Path to patient data.")
parser.add_argument('--meals_data', type=str, default='../../data/raw_data/data_files'
                                                      '/RYSA_ruokapvkirjat_0kk_1vk_nestelista_6kk_12kk_25032020_password_protected.xlsx',
                    help="Path to meals data.")
parser.add_argument('--glucose_data', type=str, default='../../data/raw_data/data_files/Operation/Operation/',
                    help="Path to glucose data.")
parser.add_argument('--processed_data', type=str, default='../../data/real/processed_data/',
                    help="Path to save processed data.")


def read_data(patient_data, meals_data):
    """Reading the encrypted data

    Parameters:
    patient_data (str): Path to patient data
    meals_data (str): Path to meals data

    Returns:
    df_patients (pd.Dataframe): Returning aggregated dataframe of patients
    df_meals (pd.Dataframe): Returning aggregated dataframe of meals
   """

    temp = io.BytesIO()
    with open(patient_data, 'rb') as f:
        excel = msoffcrypto.OfficeFile(f)
        excel.load_key('RYSA2020')
        excel.decrypt(temp)

    df_patients = pd.read_excel(temp)
    df_patients = df_patients[df_patients["Diabetes BL (no=0, yes=1)"] == 0]
    df_patients = df_patients[df_patients["ID"] != "RY14"]
    del temp

    temp = io.BytesIO()
    with open(meals_data,
              'rb') as f:
        excel = msoffcrypto.OfficeFile(f)
        excel.load_key('RYSA2020')
        excel.decrypt(temp)

    df_meals = pd.read_excel(temp)
    df_meals = df_meals.iloc[1:, :]
    df_meals = df_meals[["IvName", "DaDate", "MaTime", "STARCH", "SUGAR", "FAT"]]
    df_meals['DaDate'] = pd.to_datetime(df_meals['DaDate'], yearfirst=True)
    df_meals['MaTime'] = pd.to_datetime(df_meals['MaTime'], format='%H:%M:%S')
    df_meals["Hour"] = df_meals['MaTime'].dt.hour
    df_meals["Minute"] = df_meals['MaTime'].dt.minute
    del temp

    return df_meals, df_patients


def find_first_line(file):
    """Select first line of the file

    Parameters:
    file (str): Path to file

    Returns:
    i (int): Returning index of 1st line
   """

    i, s = 0, False
    with open(file) as f:
        for line in f:
            try:
                s = (line[0].isdigit() & line[1].isdigit())
            except:
                pass
            i += 1
            if s:
                break
    return i


def read_glucose_patient_df(patient_id, glucose_data, period='operation'):
    """Read glucose data for concrete patient

    Parameters:
    patient_id (str): id of patient
    period (str): operation or baseline
    glucose_data (str): Path to patient's glucose data

    Returns:
    df_patient_glucose (pd.Dataframe): dataframe of patient's glucose data
    df_patient_glucose['Date'].min() (datetime): start date
    df_patient_glucose['Date'].max() (datetime): end date
    False (boolean): signal value
   """

    try:
        if period == 'baseline':
            file = glob(glucose_data + "RYSA " + patient_id + " *.txt")[0]
        elif period == 'operation':
            file = glob(glucose_data + "RYSA " + patient_id + " *.txt")[0]
    except:
        return None, None, None, True

    i = find_first_line(file)

    if period == 'baseline':
        if patient_id == "57":
            i = 3

    df_patient_glucose = pd.read_csv(file, sep="\t", header=None, skiprows=i - 1, decimal=',')
    df_patient_glucose = df_patient_glucose.iloc[:, 1:5]
    df_patient_glucose.columns = ["Time", "Appointment type", "y1", "y"]
    df_patient_glucose["y"] = df_patient_glucose["y"].fillna(0.0) + df_patient_glucose["y1"].fillna(0.0)
    df_patient_glucose.drop(["y1"], inplace=True, axis=1)
    df_patient_glucose[['Date', 'Time']] = df_patient_glucose['Time'].str.split(pat=' ', n=1, expand=True)

    # Filter df_patient_glucose, select useful column
    df_patient_glucose = df_patient_glucose[df_patient_glucose['Time'].notnull()]
    df_patient_glucose = df_patient_glucose[df_patient_glucose['y'] != 0.0]

    df_patient_glucose['Time index'] = df_patient_glucose['Time'].str[:2].astype(int) * 60 + df_patient_glucose[
                                                                                                 'Time'].str[3:].astype(
        int)
    df_patient_glucose['Date'] = pd.to_datetime(df_patient_glucose['Date'], yearfirst=True)
    df_patient_glucose['Date index'] = (df_patient_glucose['Date'] - df_patient_glucose['Date'].min()).dt.days
    df_patient_glucose['t'] = df_patient_glucose['Date index'] * 1440 + df_patient_glucose['Time index']

    df_patient_glucose['id'] = patient_id

    return df_patient_glucose[["id", "t", "y"]], df_patient_glucose['Date'].min(), df_patient_glucose[
        'Date'].max(), False


def read_meals_patient_df(patient_id, date_min, date_max, df_meals):
    """Read time-interval meals data for concrete patient

    Parameters:
    patient_id (str): id of patient
    date_min (datetime): start time
    date_max (datetime): end time
    df_meals (pd.Dataframe): Aggregated dataframe of meals

    Returns:
    df_patient_meals (pd.Dataframe): dataframe of patient's meals data
    False (boolean): signal value
   """

    df_patient_meals = df_meals[df_meals['IvName'] == "RY" + patient_id.zfill(2)]
    mask = (df_patient_meals['DaDate'] >= date_min) & (df_patient_meals['DaDate'] <= date_max)
    df_patient_meals = df_patient_meals.loc[mask]

    # Checking if dataframe is empty
    if df_patient_meals.empty:
        return df_patient_meals, True

    df_patient_meals['Date index'] = (df_patient_meals['DaDate'] - date_min).dt.days
    df_patient_meals['Time index'] = df_patient_meals['Hour'] * 60 + df_patient_meals['Minute']
    df_patient_meals['t'] = df_patient_meals['Date index'] * 1440 + df_patient_meals['Time index']
    df_patient_meals['id'] = patient_id

    df_patient_meals = df_patient_meals.groupby(['id', 't']).agg(
        {'STARCH': 'sum', 'SUGAR': 'sum', 'FAT': 'sum'}).reset_index()
    df_patient_meals = df_patient_meals[(df_patient_meals["STARCH"] != 0) | (df_patient_meals["SUGAR"] != 0) | (
            df_patient_meals["FAT"] != 0)]  # Delete meals, where all nutrients are 0

    if patient_id == "46":
        df_patient_meals = df_patient_meals[df_patient_meals['t'] != 2355]

    return df_patient_meals[["id", "t", "STARCH", "SUGAR", "FAT"]], False


def create_combined_df(df_patients, df_meals, glucose_data_path, period='operation'):
    """Create combined dataframe of patient meals and glucose data

    Parameters:
    df_patients (pd.Dataframe): Aggregated dataframe of patients
    df_meals (pd.Dataframe): Aggregated dataframe of patients meals

    Returns:
    df (pd.Dataframe): dataframe of patients' meals/glucose data
   """

    patient_ids = list(df_patients['ID'].str[2:])
    df = pd.DataFrame(columns=["id", "t", "y", "STARCH", "SUGAR", "FAT"])

    for patient_id in patient_ids:
        df_patient_glucose, date_min, date_max, signal = read_glucose_patient_df(patient_id, glucose_data_path, period)

        if signal:
            continue
        df_patient_meals, signal = read_meals_patient_df(patient_id, date_min, date_max, df_meals)
        if signal:
            continue
        df_patient = pd.concat([df_patient_meals, df_patient_glucose]).sort_values(by=['t'])
        df = pd.concat([df, df_patient])

    df[["STARCH", "SUGAR", "FAT"]] = df[["STARCH", "SUGAR", "FAT"]].fillna(0.0)

    return df


def cut_meals_intervals(df):
    """Cuts combined dataframe of patient meals and glucose data into meal-day intervals

    Parameters:
    df (pd.Dataframe): dataframe of patients' meals/glucose data

    Returns:
    df1 (pd.Dataframe): newly cut dataframe of patients' meals/glucose data
   """
    patient_ids = set(df['id'])
    duration, before = 3.0 * 1440, 180
    df1 = pd.DataFrame(columns=["id", "t", "y", "STARCH", "SUGAR", "FAT"])

    for patient_id in patient_ids:
        i = 1
        df_patient = df[df['id'] == patient_id]
        meal_times = df_patient[(df_patient["STARCH"] != 0) | (df_patient["SUGAR"] != 0) | (df_patient["FAT"] != 0)][
            't'].to_numpy()

        while meal_times.any():
            first_meal_time = meal_times[0]
            last_meal_time = first_meal_time + duration
            group_meals = meal_times[meal_times < (last_meal_time)]
            meal_times = meal_times[meal_times > (last_meal_time)]

            df_patient_sliced = df_patient[
                (df_patient['t'] >= (first_meal_time - before)) & (df_patient['t'] <= (last_meal_time))]
            df_patient_sliced.loc[:, 'id'] = patient_id + '_' + str(i)
            t1 = df_patient_sliced['t'].min()

            df_patient_sliced['t'] = df_patient_sliced['t'] - t1 + 1
            df1 = pd.concat([df1, df_patient_sliced])
            i += 1

    return df1


def plot_data(df_sliced):
    """Plots data of patient meals and glucose for meal-day intervals

    Parameters:
    df_sliced (pd.Dataframe): cut dataframe of patients' meals/glucose data

    Returns:
    -
   """

    ids = set(df_sliced['id'])
    for patient_id in ids:
        df = df_sliced[df_sliced['id'] == patient_id]

        fig, axs = plt.subplots(2, 1, figsize=(20, 10))
        fig.suptitle("Glucose level, based on time, meals for patient {id}.".format(id=patient_id))

        axs[0].scatter(df['t'], df['y'], s=7, c="darkblue")
        axs[0].set(ylabel="Glucose")

        axs[1].bar(df['t'], df['STARCH'], color='darkblue', width=20)
        axs[1].bar(df['t'], df['SUGAR'], bottom=df['STARCH'], color='blue', width=20)
        axs[1].bar(df['t'], df['FAT'], bottom=df['STARCH'] + df['SUGAR'], color='slateblue', width=20)
        axs[1].set(xlabel="Time", ylabel="Stacked meals")
        axs[1].legend(['Starch', 'Sugar', 'Fat'])

        plt.show()


def train_data_save(df_sliced, path):
    """Create and save training data

    Parameters:
    path (str): path to training data folder
    df_sliced (pd.Dataframe): cut train dataframe of patients' meals/glucose data

    Returns:
    -
   """
    df_sliced['t'] = df_sliced['t'] / 60
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

        df_patient2_meals_carbs = df_patient2['STARCH'] + df_patient2[
            'SUGAR']  # Carbs meals observations for each patient
        df_patient2_meals_fat = df_patient2['FAT']  # Fat meals observations for each patient
        df_patient2_meals = df_patient2['STARCH'] + df_patient2['SUGAR'] + df_patient2[
            'FAT']  # Meals observations for each patient

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

    df_sliced['CARBS'] = df_sliced['SUGAR'] + df_sliced['STARCH']
    df_sliced.to_csv(path + "df_sliced.csv")


def test_data_save(df_sliced_test, path):
    """Create and save training data

    Parameters:
    path (str): path to training data folder
    df_sliced_test (pd.Dataframe): cut test dataframe of patients' meals/glucose data

    Returns:
    -
   """
    df_sliced_test['t'] = df_sliced_test['t'] / 60
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

        df_patient2_meals_carbs = df_patient2['STARCH'] + df_patient2[
            'SUGAR']  # Carbs meals observations for each patient
        df_patient2_meals_fat = df_patient2['FAT']  # Fat meals observations for each patient
        df_patient2_meals = df_patient2['STARCH'] + df_patient2['SUGAR'] + df_patient2[
            'FAT']  # Meals observations for each patient

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

    df_sliced_test['CARBS'] = df_sliced_test['SUGAR'] + df_sliced_test['STARCH']
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

    return 0


if __name__ == "__main__":
    args = parser.parse_args()

    # Create combined meals/glucose dataframe
    df_meals, df_patients = read_data(args.patient_data, args.meals_data)
    data = create_combined_df(df_patients, df_meals, args.glucose_data, args.period)
    df_sliced = cut_meals_intervals(data)

    # Select not noisy patient data
    # plot_data(df_sliced) # If there is need to select patients
    if args.period == 'operation':
        good_ids = ['31_2', '12_2', '32_2', '46_2', '29_2', '57_2', '23_2', '9_2', '28_2', '76_1', '65_1', '60_1']
    elif args.period == 'baseline':
        good_ids = ['31_1', '12_1', '32_1', '46_1', '29_1', '57_1', '23_1', '9_1', '28_1', '76_1', '65_1', '60_1']
    df_sliced = df_sliced[df_sliced['id'].isin(good_ids)]

    # Median trend for each patient
    if args.period == 'operation':
        trend_p_dict = {'12_2': 3.5, '32_2': 3.6, '31_2': 3.7, '46_2': 3.6, '29_2': 3.7, '57_2': 4.1, '23_2': 4.7,
                        '9_2': 4.0, '28_2': 3.4, '76_1': 4.4, '65_1': 4.8, '60_1': 4.5}
    elif args.period == 'baseline':
        trend_p_dict = {'31_1': 4.5, '12_1': 5.2, '32_1': 4.9, '46_1': 4.0, '29_1': 4.4, '57_1': 5.6, '23_1': 5.9,
                        '9_1': 4.7, '28_1': 4.3, '76_1': 5.8, '65_1': 6.2, '60_1': 5.1}

    # Divide data into train/test
    df_sliced_test = df_sliced[df_sliced['t'] >= 2600]
    df_sliced = df_sliced[df_sliced['t'] <= 2600]

    # Save preprocessed training data
    train_data_save(df_sliced, args.processed_data)
    test_data_save(df_sliced_test, args.processed_data)

    # Combine train and test data
    combine_train_test(args.processed_data)
