import numpy as np
import os
import datetime
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams, colors

# customized settings
params = {  # 'backend': 'ps',
    'font.family': 'serif',
    'font.serif': 'Latin Modern Roman',
    'font.size': 10,
    'axes.labelsize': 'medium',
    'axes.titlesize': 'medium',
    'legend.fontsize': 'medium',
    'xtick.labelsize': 'small',
    'ytick.labelsize': 'small',
    'savefig.dpi': 150,
    'text.usetex': True}
# tell matplotlib about your params
rcParams.update(params)

# set nice figure sizes
fig_width_pt = 245    # Get this from LaTeX using \showthe\columnwidth
golden_mean = (np.sqrt(5.) - 1.) / 2.  # Aesthetic ratio
ratio = golden_mean
inches_per_pt = 1. / 72.27  # Convert pt to inches
fig_width = fig_width_pt * inches_per_pt  # width in inches
fig_height = fig_width*ratio  # height in inches
fig_size = [2*fig_width, 2*fig_height]
rcParams.update({'figure.figsize': fig_size})

def extract_forecasts(data, forecast_period):
    
    dates = []
    data_quantile_0025_cont = []
    data_quantile_0975_cont = []
    data_point_cont = []
    
    for file in sorted(os.listdir(ensemble_forecast_directory))[7:]:    

        if file[-4:] == '.csv':
            
            data = pd.read_csv(ensemble_forecast_directory+file)
            
            data = data[data['location'] == 'US']
                
            data_quantile = data[data['type'] == 'quantile']
                        
            data_quantile_0025 = data[data['quantile'] == 0.025]    
            data_quantile_0975 = data[data['quantile'] == 0.975]     
            
            data_point = data[data['type'] == 'point']
            
            dates.append(data_quantile_0025[data_quantile_0025['target'] == forecast_period]['target_end_date'].to_numpy()[0])
            data_quantile_0025_cont.append(float(data_quantile_0025[data_quantile_0025['target'] == forecast_period]['value']))
            data_quantile_0975_cont.append(float(data_quantile_0975[data_quantile_0975['target'] == forecast_period]['value']))
            data_point_cont.append(float(data_point[data_point['target'] == forecast_period]['value']))
    
    dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]
    
    return np.asarray(dates), np.asarray(data_quantile_0025_cont), np.asarray(data_quantile_0975_cont), np.asarray(data_point_cont)

covid_data = pd.read_csv('../../data/time_series_covid19_deaths_US.csv')

colnames = covid_data.columns.tolist()

start = datetime.datetime.strptime(colnames[12], "%m/%d/%y")
end = datetime.datetime.strptime(colnames[-1], "%m/%d/%y")
dates_covid_data = np.asarray([start + datetime.timedelta(days=x) for x in range(0, (end-start).days)])
dates_covid_data = dates_covid_data[3:][::7]
deaths_covid_data = [covid_data.iloc[:,i].sum() for i in range(12,len(colnames))]
deaths_covid_data = np.diff(deaths_covid_data[3:])
deaths_covid_data = np.add.reduceat(deaths_covid_data, np.arange(0, len(deaths_covid_data), 7))

ensemble_forecast_directory = '../../data/COVIDhub-ensemble/'
    
dates_1wk, data_quantile_0025_1wk, data_quantile_0975_1wk, data_point_1wk = \
extract_forecasts(ensemble_forecast_directory, '1 wk ahead inc death')

dates_2wk, data_quantile_0025_2wk, data_quantile_0975_2wk, data_point_2wk = \
extract_forecasts(ensemble_forecast_directory, '2 wk ahead inc death')

dates_3wk, data_quantile_0025_3wk, data_quantile_0975_3wk, data_point_3wk = \
extract_forecasts(ensemble_forecast_directory, '3 wk ahead inc death')

dates_4wk, data_quantile_0025_4wk, data_quantile_0975_4wk, data_point_4wk = \
extract_forecasts(ensemble_forecast_directory, '4 wk ahead inc death')

div = 1e3

def error_comparison(deaths_covid_data, data_point, dates, dates_covid_data, weeks):
    error_euler = []
    error_ensemble = []
    
    for i in range(len(dates)):
        
        ind = np.where(dates_covid_data == dates[i])
            
        if len(ind[0]):
            ind = ind[0][0]
            error_euler.append(abs(deaths_covid_data[ind-weeks]-deaths_covid_data[ind]))
            error_ensemble.append(abs(data_point[i]-deaths_covid_data[ind]))
      
    error_euler = np.asarray(error_euler)/div  
    error_ensemble = np.asarray(error_ensemble)/div    
    error_euler_cum = np.cumsum(error_euler)
    error_ensemble_cum = np.cumsum(error_ensemble)
    
    return error_euler, error_ensemble, error_euler_cum, error_ensemble_cum

error_euler_1wk, error_ensemble_1wk, error_euler_1wk_cum, error_ensemble_1wk_cum = \
error_comparison(deaths_covid_data, data_point_1wk, dates_1wk, dates_covid_data, 1)

error_euler_2wk, error_ensemble_2wk, error_euler_2wk_cum, error_ensemble_2wk_cum = \
error_comparison(deaths_covid_data, data_point_2wk, dates_2wk, dates_covid_data, 2)

error_euler_3wk, error_ensemble_3wk, error_euler_3wk_cum, error_ensemble_3wk_cum = \
error_comparison(deaths_covid_data, data_point_3wk, dates_3wk, dates_covid_data, 3)

error_euler_4wk, error_ensemble_4wk, error_euler_4wk_cum, error_ensemble_4wk_cum = \
error_comparison(deaths_covid_data, data_point_4wk, dates_4wk, dates_covid_data, 4)

fig, ax = plt.subplots(ncols = 2, nrows = 2)

#ax[0,0].set_title(r'1 week forecast')
ax[0,0].text(dates_1wk[0]-datetime.timedelta(days=7), 0.9*60, r'(a)')
ax[0,0].plot(dates_1wk[:-2],error_euler_1wk[:-1], color = 'tab:red', linewidth = 1.5) 
ax[0,0].plot(dates_1wk[:-2],error_euler_1wk_cum[:-1], color = 'tab:red', ls = (0, (5, 1)), linewidth = 1.5) 
ax[0,0].plot(dates_1wk[:-2],error_ensemble_1wk[:-1], color = 'tab:blue', linewidth = 1.5)  
ax[0,0].plot(dates_1wk[:-2],error_ensemble_1wk_cum[:-1], color = 'tab:blue', ls = (0, (5, 1)), linewidth = 1.5)  
ax[0,0].set_xlim([datetime.date(2020, 6, 1), datetime.date(2021, 3, 1)])
ax[0,0].set_xticks([])
ax[0,0].set_ylabel(r'Prediction error [$\times 10^3$]')
ax[0,0].set_ylim(0,60)

#ax[0,1].set_title(r'2 week forecast')
ax[0,1].text(dates_1wk[0]-datetime.timedelta(days=7), 0.9*60, r'(b)')
ax[0,1].plot(dates_2wk[:-3],error_euler_2wk[:-1], color = 'tab:red', linewidth = 1.5) 
ax[0,1].plot(dates_2wk[:-3],error_euler_2wk_cum[:-1], color = 'tab:red', ls = (0, (5, 1)), linewidth = 1.5) 
ax[0,1].plot(dates_2wk[:-3],error_ensemble_2wk[:-1], color = 'tab:blue', linewidth = 1.5)  
ax[0,1].plot(dates_2wk[:-3],error_ensemble_2wk_cum[:-1], color = 'tab:blue', ls = (0, (5, 1)), linewidth = 1.5)  
ax[0,1].set_xlim([datetime.date(2020, 6, 1), datetime.date(2021, 3, 1)])
ax[0,1].set_xticks([])
ax[0,1].set_ylim(0,60)
ax[0,1].set_yticks([])

#ax[1,0].set_title(r'3 week forecast')
ax[1,0].text(dates_1wk[0]-datetime.timedelta(days=7), 0.9*100, r'(c)')
ax[1,0].plot(dates_3wk[:-4],error_euler_3wk[:-1], color = 'tab:red', linewidth = 1.5) 
ax[1,0].plot(dates_3wk[:-4],error_euler_3wk_cum[:-1], color = 'tab:red', ls = (0, (5, 1)), linewidth = 1.5) 
ax[1,0].plot(dates_3wk[:-4],error_ensemble_3wk[:-1], color = 'tab:blue', linewidth = 1.5)  
ax[1,0].plot(dates_3wk[:-4],error_ensemble_3wk_cum[:-1], color = 'tab:blue', ls = (0, (5, 1)), linewidth = 1.5)  
ax[1,0].set_xlim([datetime.date(2020, 6, 1), datetime.date(2021, 3, 1)])
ax[1,0].tick_params(axis='x', labelrotation=45)
ax[1,0].set_ylabel(r'Prediction error [$\times 10^3$]')
ax[1,0].set_ylim(0,100)

#ax[1,1].set_title(r'4 week forecast')
ax[1,1].text(dates_1wk[0]-datetime.timedelta(days=7), 0.9*100, r'(d)')
ax[1,1].plot(dates_4wk[:-5],error_euler_4wk[:-1], color = 'tab:red', linewidth = 1.5) 
ax[1,1].plot(dates_4wk[:-5],error_euler_4wk_cum[:-1], color = 'tab:red', ls = (0, (5, 1)), linewidth = 1.5) 
ax[1,1].plot(dates_4wk[:-5],error_ensemble_4wk[:-1], color = 'tab:blue', linewidth = 1.5)  
ax[1,1].plot(dates_4wk[:-5],error_ensemble_4wk_cum[:-1], color = 'tab:blue', ls = (0, (5, 1)), linewidth = 1.5)  
ax[1,1].set_xlim([datetime.date(2020, 6, 1), datetime.date(2021, 3, 1)])
ax[1,1].tick_params(axis='x', labelrotation=45)
ax[1,1].set_ylim(0,100)
ax[1,1].set_yticks([])

plt.tight_layout()
plt.savefig('prediction_error.png', dpi = 480)

# plot
fig, ax = plt.subplots(ncols = 2, nrows = 2)
# for cumulative numbers, use np.cumsum(deaths_covid_data)

#ax[0,0].set_title(r'1 week forecast')
ax[0,0].text(dates_1wk[0]-datetime.timedelta(days=7), 0.9*30, r'(a)')
ax[0,0].fill_between(dates_1wk, data_quantile_0975_1wk/div, data_quantile_0025_1wk/div, facecolor = 'tab:blue', alpha = 0.5)
ax[0,0].plot(dates_1wk, data_point_1wk/div, color = 'tab:blue', linewidth = 1.5, label = r'CDC ensemble')
ax[0,0].plot(dates_covid_data+datetime.timedelta(days=5), deaths_covid_data/div, color = 'tab:red', linewidth = 1.5, label = r'Euler')
ax[0,0].plot(dates_covid_data, deaths_covid_data/div, ls = (0, (5, 1)), color = 'k', linewidth = 1.5, label = r'reported')

#ax[0,0].plot(dates_covid_data[1:], [deaths_covid_data[i]/div+(deaths_covid_data[i]-deaths_covid_data[i-1])/div for i in range(1,len(deaths_covid_data))], "orange")

ax[0,0].set_xlim([datetime.date(2020, 6, 1), datetime.date(2021, 3, 1)])
ax[0,0].set_xticks([])
ax[0,0].set_ylabel(r'Weekly deaths [$\times 10^3$]')
ax[0,0].set_ylim(0,30)

#ax[0,1].set_title(r'2 week forecast')
ax[0,1].text(dates_1wk[0]-datetime.timedelta(days=7), 0.9*30, r'(b)')
ax[0,1].fill_between(dates_2wk, data_quantile_0975_2wk/div, data_quantile_0025_2wk/div, facecolor = 'tab:blue', alpha = 0.5)
ax[0,1].plot(dates_2wk, data_point_2wk/div, color = 'tab:blue', linewidth = 1.5)
ax[0,1].plot(dates_covid_data+datetime.timedelta(days=12), deaths_covid_data/div, color = 'tab:red', linewidth = 1.5)
ax[0,1].plot(dates_covid_data, deaths_covid_data/div, ls = (0, (5, 1)), color = 'k', linewidth = 1.5)
ax[0,1].set_xlim([datetime.date(2020, 6, 1), datetime.date(2021, 3, 1)])
ax[0,1].set_xticks([])
ax[0,1].set_ylim(0,30)
ax[0,1].set_yticks([])

#ax[1,0].set_title(r'3 week forecast')
ax[1,0].text(dates_1wk[0]-datetime.timedelta(days=7), 0.9*30, r'(c)')
ax[1,0].fill_between(dates_3wk, data_quantile_0975_3wk/div, data_quantile_0025_3wk/div, facecolor = 'tab:blue', alpha = 0.5)
ax[1,0].plot(dates_3wk, data_point_3wk/div, color = 'tab:blue', linewidth = 1.5)
ax[1,0].plot(dates_covid_data+datetime.timedelta(days=19), deaths_covid_data/div, color = 'tab:red', linewidth = 1.5)
ax[1,0].plot(dates_covid_data, deaths_covid_data/div, ls = (0, (5, 1)), color = 'k', linewidth = 1.5)
ax[1,0].set_xlim([datetime.date(2020, 6, 1), datetime.date(2021, 3, 1)])
ax[1,0].tick_params(axis='x', labelrotation=45)
ax[1,0].set_ylabel(r'Weekly deaths [$\times 10^3$]')
ax[1,0].set_ylim(0,30)

#ax[1,1].set_title(r'4 week forecast')
ax[1,1].text(dates_1wk[0]-datetime.timedelta(days=7), 0.9*30, r'(d)')
ax[1,1].fill_between(dates_4wk, data_quantile_0975_4wk/div, data_quantile_0025_4wk/div, facecolor = 'tab:blue', alpha = 0.5)
ax[1,1].plot(dates_4wk, data_point_4wk/div, color = 'tab:blue', linewidth = 1.5)
ax[1,1].plot(dates_covid_data+datetime.timedelta(days=26), deaths_covid_data/div, color = 'tab:red', linewidth = 1.5)
ax[1,1].plot(dates_covid_data, deaths_covid_data/div, ls = (0, (5, 1)), color = 'k', linewidth = 1.5)

#ax[1,1].plot(dates_covid_data[4:], [deaths_covid_data[i]/div+4*(deaths_covid_data[i]-deaths_covid_data[i-1])/div for i in range(len(deaths_covid_data)-4)], "orange")

ax[1,1].set_xlim([datetime.date(2020, 6, 1), datetime.date(2021, 3, 1)])
ax[1,1].tick_params(axis='x', labelrotation=45)
ax[1,1].set_ylim(0,30)
ax[1,1].set_yticks([])
          
plt.tight_layout()

plt.savefig("ensemble_forecast.png", dpi = 480)
        
