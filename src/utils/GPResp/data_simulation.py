import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def simulate_artificial_data():
    # Artificial data
    n = 10
    t0, tn = 0.0, 40.0
    patients = ['1_1','2_1','3_1','4_1','5_1','6_1','7_1','8_1','9_1','10_1']
    trend = [3.5,3.6,3.7,3.5,3.2,4.0,3.3,3.1,3.5,3.4]
    glucose_num = [160, 150, 120, 130, 175, 165, 155, 125, 135, 170]
    meals = [np.array([[8.0,10,7],[12.5,12,3],[17.5,8,8],[20.0,8,5],[28.0,10,10],[32.5,9,8]]),
             np.array(
                 [[6.5, 12, 7], [12.0, 11, 3], [16.0, 8, 8], [20.0, 8, 5], [28.0, 10, 10], [32.5, 11, 2]]),
             np.array(
                 [[8.0, 9, 7], [12.5, 12, 6], [17.0, 8, 8], [20.5, 10, 5], [29.0, 10, 10], [32.5, 11, 7]]),
             np.array(
                 [[7.0, 6, 7], [13.5, 12, 8], [16.0, 8, 8], [20.0, 15, 5.5], [29.0, 10, 10], [32.5, 6, 7]]),
             np.array(
                 [[8.0, 6, 4], [12.5, 10, 5], [17.0, 8, 8], [20.0, 8, 12], [27.0, 10, 10], [33.5, 15, 6]]),
             np.array(
                 [[9.0, 10, 3], [13.5, 8, 3], [16.5, 8, 8], [20.5, 8, 5], [28.0, 10, 10], [32.5, 11, 2]]),
             np.array(
                 [[8.0, 10, 12], [13.0, 12, 3], [16.0, 8, 8], [20.0, 8, 6], [28.0, 10, 10], [33.5, 11, 2]]),
             np.array(
                 [[8.0, 10, 5], [13.5, 12, 3], [16.0, 8, 8], [20.5, 12, 7], [28.0, 10, 10], [32.5, 10, 4]]),
             np.array(
                 [[8.5, 10, 4], [12.5, 12, 3], [16.0, 8, 8], [20.0, 10, 5], [28.0, 10, 10], [33.5, 11, 7]]),
             np.array(
                 [[8.0, 10, 7], [14.0, 12, 3], [16.0, 8, 8], [20.5, 8, 5], [28.0, 10, 10], [32.5, 9, 2]])]

    b_carbs, b_fat = [0.06,0.065,0.05,0.065,0.06,0.05,0.06,0.06,0.07,0.06], [0.08,0.07,0.09,0.095,0.095,0.07,0.065,0.07,0.07,0.08,]
    l = [0.5,0.4,0.55,0.5,0.6,0.57,0.5,0.6,0.45,0.6]

    # Save to train/test dataframe
    df = pd.DataFrame(columns=['id','t','y','CARBS','FAT'])

    # Simulate artificial glucose values
    for p in range(n):
        # Save meals data to dataframe
        for m in meals[p]:
            df = df.append({'id':patients[p],'t':m[0],'y':np.nan,'CARBS':m[1],'FAT':m[2]}, ignore_index=True)

        # Save glucose data to dataframe
        glucose_times = np.linspace(t0,tn,glucose_num[p])
        glucose = []
        for t in glucose_times:
            sum_meals = 0
            for m in meals[p]:
                if t>m[0] and m[0]<=(t+4.0):
                    sum_meals += (b_carbs[p]*m[1]+b_fat[p]*m[2]) * np.exp((-0.5)*((t-m[0]-3*l[p]))**2/(l[p]**2))
            sum_meals += trend[p]
            glucose.append(sum_meals)

            df = df.append({'id': patients[p], 't': t, 'y': sum_meals, 'CARBS': 0.0, 'FAT': 0.0}, ignore_index=True)

        #plt.plot(glucose_times, glucose, c="midnightblue", linewidth=2, label='True glucose curve', zorder=2)
        #plt.plot(glucose_times, glucose, 'kx', ms=5, alpha=0.5, label='True observations')
        #plt.show()

    df.sort_values(by=['id', 't'], inplace=True)
    df_train = df[df['t'] <= 25]
    df_test = df[df['t'] > 25]

    return df_train, df_test

