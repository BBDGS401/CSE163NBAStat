import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score


def initialize_dataset(df):
    df = df.iloc[:, [0, 1, 5, 11, 12, 13, 43, 44, 45, 46, 47, 48, 49, 50, 51,
                     64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77,
                     78, 79, 80, 81, 82, 83, 84]]
    df['CRFGM'] = df['RAFGM'] + df['NRAFGM']
    df['CRFGA'] = df['RAFGA'] + df['NRAFGA']
    df['CRFG%'] = round(df['CRFGM'] / df['CRFGA'] * 100, 1)
    df['OCRFGM'] = df['ORAFGM'] + df['ONRAFGM']
    df['OCRFGA'] = df['ORAFGA'] + df['ONRAFGA']
    df['OCRFG%'] = round(df['OCRFGM'] / df['OCRFGA'] * 100, 1)
    df['O3PM'] = df['OC3M'] + df['OATB3M']
    df['O3PA'] = df['OC3A'] + df['OATB3A']
    df['O3P%'] = round(df['O3PM'] / df['O3PA'] * 100, 1)
    df['TEAMY'] = df['YEAR'].astype('str') + ' ' + df['TEAM']
    df = df.loc[:, ['YEAR', 'TEAMY', 'WIN%', 'CRFGM', 'CRFGA', 'CRFG%',
                    'MRFGM', 'MRFGA', 'MRFG%', '3PM', '3PA', '3P%', 'OCRFGM',
                    'OCRFGA', 'OCRFG%', 'OMRFGM', 'OMRFGA', 'OMRFG%', 'O3PM',
                    'O3PA', 'O3P%']]
    return df


def modelling(df):
    x_var = df.loc[:, ['3P%', 'CRFG%', 'O3P%', 'OCRFG%']]
    y_var = df['WIN%']
    x_train, x_test, y_train, y_test = train_test_split(x_var, y_var,
                                                        test_size=0.2)
    # After trying max_depth from 1 to 50 for many times, it seems that
    # max_depth = 5 optimizes the accuracy of the model.
    # The MAE to predict win% is around 0.06 to 0.10.
    mae_arr = []
    for i in range(1, 50):
        model = DecisionTreeRegressor(max_depth=i)
        model.fit(x_train, y_train)
        train_predictions = model.predict(x_train)
        train_mae = mean_absolute_error(y_train, train_predictions)
        test_predictions = model.predict(x_test)
        test_mae = mean_absolute_error(y_test, test_predictions)
        train_r2 = r2_score(y_train, train_predictions)
        test_r2 = r2_score(y_test, test_predictions)
        mae_arr.append({'Max depth': i, 'Train MAE': train_mae,
                        'Test MAE': test_mae, 'Train r2': train_r2,
                        'Test r2': test_r2})
    mae_arr = pd.DataFrame(mae_arr)
    print(mae_arr)


def plot_indep_var(df):
    dep_var = 'WIN%'
    for indep_var in df:
        if indep_var != 'YEAR' and indep_var != 'TEAMY' \
           and indep_var != 'WIN%':
            plot = sns.lmplot(x=indep_var, y=dep_var, col='YEAR', col_wrap=3,
                              hue='YEAR', data=df)
            plot.set(xlabel=indep_var, ylabel=dep_var)
            plot.savefig('plots/' + indep_var + '.png')


def print_r2(rowsno, df):
    r2dict = {
        'CRFGA': [], 'CRFGM': [], 'CRFG%': [], 'MRFGA': [], 'MRFGM': [],
        'MRFG%': [], '3PA': [], '3PM': [], '3P%': [], 'OCRFGA': [],
        'OCRFGM': [], 'OCRFG%': [], 'OMRFGA': [], 'OMRFGM': [], 'OMRFG%': [],
        'O3PA': [], 'O3PM': [], 'O3P%': [], 'YEAR': []
    }
    years = df['YEAR'].unique()
    rows_per_year = rowsno / len(years)
    print(rows_per_year)
    for year in reversed(years):
        filtered_df = df[df['YEAR'] == year]
        r2dict['YEAR'].append(year)
        for x in filtered_df:
            if x != 'YEAR' and x != 'TEAMY' and x != 'WIN%':
                r_square = round(cal_r2(rows_per_year, x, 'WIN%', filtered_df),
                                 3)
                print('R2 for WIN%' + ' against ' + x + ' in ' + str(year) +
                      ': ' + str(r_square))
                r2dict[x].append(r_square)
    return r2dict


def cal_r2(rowsno, x, y, df):
    slope = (rowsno * ((df[x] * df[y]).sum()) - (df[x].sum()) *
             (df[y].sum())) / (rowsno * ((df[x] ** 2).sum()) -
                               (df[x].sum()) ** 2)
    y_intercept = (df[y].sum() - slope * (df[x].sum())) / rowsno
    y_pred = y + '_pred'
    df[y_pred] = slope * df[x] + y_intercept
    y_mean = df[y].mean()
    r_square = 1 - (((df[y] - df[y_pred]) ** 2).sum()) / (((df[y] - y_mean)
                                                           ** 2).sum())
    return r_square


def graph_r2(r2dict, num):
    r2df = pd.DataFrame(r2dict)
    print(r2df)
    fig1, ax1 = plt.subplots(1)
    fig2, ax2 = plt.subplots(1)
    for indep_var in r2df:
        if indep_var != 'YEAR' and indep_var[0] != 'O':
            r2df.plot(ax=ax1, x='YEAR', y=indep_var)
        if indep_var != 'YEAR' and indep_var[0] == 'O':
            r2df.plot(ax=ax2, x='YEAR', y=indep_var)
    ax1.set_xlabel('Year')
    ax2.set_xlabel('Year')
    ax1.set_ylabel('R Squared Value')
    ax2.set_ylabel('R Squared Value')
    ax1.set_title('Offensive Shooting Stats R Squared Value From 2017 to 2022')
    ax2.set_title('Defensive Shooting Stats R Squared Value From 2017 to 2022')
    fig1.savefig('plots/r2comparison1 ' + str(num) + '.png')
    fig2.savefig('plots/r2comparison2 ' + str(num) + '.png')


def main():
    sns.set()
    df = pd.read_csv('NBA stats.csv')
    df = initialize_dataset(df)
    modelling(df)
    plot_indep_var(df)
    r2dict = print_r2(df['TEAMY'].count(), df)
    graph_r2(r2dict, 1)
    # testing my result using playoff stats
    df2 = pd.read_csv('NBA stats.csv')
    teams6 = (df2['TEAM'] == 'Golden State Warriors') |\
             (df2['TEAM'] == 'Houston Rockets') |\
             (df2['TEAM'] == 'Detroit Pistons') |\
             (df2['TEAM'] == 'Brooklyn Nets') |\
             (df2['TEAM'] == 'New York Knicks') |\
             (df2['TEAM'] == 'New Orleans Pelicans') |\
             (df2['TEAM'] == 'Milwaukee Bucks') |\
             (df2['TEAM'] == 'Sacramento Kings') |\
             (df2['TEAM'] == 'Phoenix Suns') |\
             (df2['TEAM'] == 'Boston Celtics')
    df2 = df2[teams6]
    df2 = initialize_dataset(df2)
    r2dict2 = print_r2(df2['TEAMY'].count(), df2)
    graph_r2(r2dict2, 2)


if __name__ == '__main__':
    main()
