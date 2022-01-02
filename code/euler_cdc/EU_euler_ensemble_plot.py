import numpy as np
import re
import os
import datetime
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams, colors
from  matplotlib.colors import LinearSegmentedColormap

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

ISO3166 = {
	'AD': 'Andorra',
	'AE': 'United Arab Emirates',
	'AF': 'Afghanistan',
	'AG': 'Antigua & Barbuda',
	'AI': 'Anguilla',
	'AL': 'Albania',
	'AM': 'Armenia',
	'AN': 'Netherlands Antilles',
	'AO': 'Angola',
	'AQ': 'Antarctica',
	'AR': 'Argentina',
	'AS': 'American Samoa',
	'AT': 'Austria',
	'AU': 'Australia',
	'AW': 'Aruba',
	'AZ': 'Azerbaijan',
	'BA': 'Bosnia and Herzegovina',
	'BB': 'Barbados',
	'BD': 'Bangladesh',
	'BE': 'Belgium',
	'BF': 'Burkina Faso',
	'BG': 'Bulgaria',
	'BH': 'Bahrain',
	'BI': 'Burundi',
	'BJ': 'Benin',
	'BM': 'Bermuda',
	'BN': 'Brunei Darussalam',
	'BO': 'Bolivia',
	'BR': 'Brazil',
	'BS': 'Bahama',
	'BT': 'Bhutan',
	'BU': 'Burma (no longer exists)',
	'BV': 'Bouvet Island',
	'BW': 'Botswana',
	'BY': 'Belarus',
	'BZ': 'Belize',
	'CA': 'Canada',
	'CC': 'Cocos (Keeling) Islands',
	'CF': 'Central African Republic',
	'CG': 'Congo',
	'CH': 'Switzerland',
	'CI': 'Côte D\'ivoire (Ivory Coast)',
	'CK': 'Cook Iislands',
	'CL': 'Chile',
	'CM': 'Cameroon',
	'CN': 'China',
	'CO': 'Colombia',
	'CR': 'Costa Rica',
	'CS': 'Czechoslovakia (no longer exists)',
	'CU': 'Cuba',
	'CV': 'Cape Verde',
	'CX': 'Christmas Island',
	'CY': 'Cyprus',
	'CZ': 'Czech Republic',
	'DD': 'German Democratic Republic (no longer exists)',
	'DE': 'Germany',
	'DJ': 'Djibouti',
	'DK': 'Denmark',
	'DM': 'Dominica',
	'DO': 'Dominican Republic',
	'DZ': 'Algeria',
	'EC': 'Ecuador',
	'EE': 'Estonia',
	'EG': 'Egypt',
	'EH': 'Western Sahara',
	'ER': 'Eritrea',
	'ES': 'Spain',
	'ET': 'Ethiopia',
	'FI': 'Finland',
	'FJ': 'Fiji',
	'FK': 'Falkland Islands (Malvinas)',
	'FM': 'Micronesia',
	'FO': 'Faroe Islands',
	'FR': 'France',
	'FX': 'France, Metropolitan',
	'GA': 'Gabon',
	'GB': 'United Kingdom',
	'GD': 'Grenada',
	'GE': 'Georgia',
	'GF': 'French Guiana',
	'GH': 'Ghana',
	'GI': 'Gibraltar',
	'GL': 'Greenland',
	'GM': 'Gambia',
	'GN': 'Guinea',
	'GP': 'Guadeloupe',
	'GQ': 'Equatorial Guinea',
	'GR': 'Greece',
	'GS': 'South Georgia and the South Sandwich Islands',
	'GT': 'Guatemala',
	'GU': 'Guam',
	'GW': 'Guinea-Bissau',
	'GY': 'Guyana',
	'HK': 'Hong Kong',
	'HM': 'Heard & McDonald Islands',
	'HN': 'Honduras',
	'HR': 'Croatia',
	'HT': 'Haiti',
	'HU': 'Hungary',
	'ID': 'Indonesia',
	'IE': 'Ireland',
	'IL': 'Israel',
	'IN': 'India',
	'IO': 'British Indian Ocean Territory',
	'IQ': 'Iraq',
	'IR': 'Islamic Republic of Iran',
	'IS': 'Iceland',
	'IT': 'Italy',
	'JM': 'Jamaica',
	'JO': 'Jordan',
	'JP': 'Japan',
	'KE': 'Kenya',
	'KG': 'Kyrgyzstan',
	'KH': 'Cambodia',
	'KI': 'Kiribati',
	'KM': 'Comoros',
	'KN': 'St. Kitts and Nevis',
	'KP': 'Korea, Democratic People\'s Republic of',
	'KR': 'Korea, Republic of',
	'KW': 'Kuwait',
	'KY': 'Cayman Islands',
	'KZ': 'Kazakhstan',
	'LA': 'Lao People\'s Democratic Republic',
	'LB': 'Lebanon',
	'LC': 'Saint Lucia',
	'LI': 'Liechtenstein',
	'LK': 'Sri Lanka',
	'LR': 'Liberia',
	'LS': 'Lesotho',
	'LT': 'Lithuania',
	'LU': 'Luxembourg',
	'LV': 'Latvia',
	'LY': 'Libyan Arab Jamahiriya',
	'MA': 'Morocco',
	'MC': 'Monaco',
	'MD': 'Moldova, Republic of',
	'MG': 'Madagascar',
	'MH': 'Marshall Islands',
	'ML': 'Mali',
	'MN': 'Mongolia',
	'MM': 'Myanmar',
	'MO': 'Macau',
	'MP': 'Northern Mariana Islands',
	'MQ': 'Martinique',
	'MR': 'Mauritania',
	'MS': 'Monserrat',
	'MT': 'Malta',
	'MU': 'Mauritius',
	'MV': 'Maldives',
	'MW': 'Malawi',
	'MX': 'Mexico',
	'MY': 'Malaysia',
	'MZ': 'Mozambique',
	'NA': 'Namibia',
	'NC': 'New Caledonia',
	'NE': 'Niger',
	'NF': 'Norfolk Island',
	'NG': 'Nigeria',
	'NI': 'Nicaragua',
	'NL': 'Netherlands',
	'NO': 'Norway',
	'NP': 'Nepal',
	'NR': 'Nauru',
	'NT': 'Neutral Zone (no longer exists)',
	'NU': 'Niue',
	'NZ': 'New Zealand',
	'OM': 'Oman',
	'PA': 'Panama',
	'PE': 'Peru',
	'PF': 'French Polynesia',
	'PG': 'Papua New Guinea',
	'PH': 'Philippines',
	'PK': 'Pakistan',
	'PL': 'Poland',
	'PM': 'St. Pierre & Miquelon',
	'PN': 'Pitcairn',
	'PR': 'Puerto Rico',
	'PT': 'Portugal',
	'PW': 'Palau',
	'PY': 'Paraguay',
	'QA': 'Qatar',
	'RE': 'Réunion',
	'RO': 'Romania',
	'RU': 'Russian Federation',
	'RW': 'Rwanda',
	'SA': 'Saudi Arabia',
	'SB': 'Solomon Islands',
	'SC': 'Seychelles',
	'SD': 'Sudan',
	'SE': 'Sweden',
	'SG': 'Singapore',
	'SH': 'St. Helena',
	'SI': 'Slovenia',
	'SJ': 'Svalbard & Jan Mayen Islands',
	'SK': 'Slovakia',
	'SL': 'Sierra Leone',
	'SM': 'San Marino',
	'SN': 'Senegal',
	'SO': 'Somalia',
	'SR': 'Suriname',
	'ST': 'Sao Tome & Principe',
	'SU': 'Union of Soviet Socialist Republics (no longer exists)',
	'SV': 'El Salvador',
	'SY': 'Syrian Arab Republic',
	'SZ': 'Swaziland',
	'TC': 'Turks & Caicos Islands',
	'TD': 'Chad',
	'TF': 'French Southern Territories',
	'TG': 'Togo',
	'TH': 'Thailand',
	'TJ': 'Tajikistan',
	'TK': 'Tokelau',
	'TM': 'Turkmenistan',
	'TN': 'Tunisia',
	'TO': 'Tonga',
	'TP': 'East Timor',
	'TR': 'Turkey',
	'TT': 'Trinidad & Tobago',
	'TV': 'Tuvalu',
	'TW': 'Taiwan, Province of China',
	'TZ': 'Tanzania, United Republic of',
	'UA': 'Ukraine',
	'UG': 'Uganda',
	'UM': 'United States Minor Outlying Islands',
	'US': 'United States of America',
	'UY': 'Uruguay',
	'UZ': 'Uzbekistan',
	'VA': 'Vatican City State (Holy See)',
	'VC': 'St. Vincent & the Grenadines',
	'VE': 'Venezuela',
	'VG': 'British Virgin Islands',
	'VI': 'United States Virgin Islands',
	'VN': 'Viet Nam',
	'VU': 'Vanuatu',
	'WF': 'Wallis & Futuna Islands',
	'WS': 'Samoa',
	'YD': 'Democratic Yemen (no longer exists)',
	'YE': 'Yemen',
	'YT': 'Mayotte',
	'YU': 'Yugoslavia',
	'ZA': 'South Africa',
	'ZM': 'Zambia',
	'ZR': 'Zaire',
	'ZW': 'Zimbabwe',
	'ZZ': 'Unknown or unspecified country'}

def extract_forecasts(data, forecast_period):
    
    dates = []
    forecast_dates = []
    data_quantile_0025_cont = []
    data_quantile_0975_cont = []
    data_point_cont = []
    
    for file in sorted(os.listdir(ensemble_forecast_directory))[7:]:    

        if file[-4:] == '.csv':
            
            data = pd.read_csv(ensemble_forecast_directory+file)
                            
            data_quantile = data[data['type'] == 'quantile']
                        
            data_quantile_0025 = data[data['quantile'] == 0.025]    
            data_quantile_0975 = data[data['quantile'] == 0.975]     
            
            data_point = data[data['type'] == 'point']
                        
            forecast_dates.append(data_quantile_0025[data_quantile_0025['target'] == forecast_period]['forecast_date'].to_numpy()[0])
            dates.append(data_quantile_0025[data_quantile_0025['target'] == forecast_period]['target_end_date'].to_numpy()[0])
            data_quantile_0025_cont.append(sum(data_quantile_0025[data_quantile_0025['target'] == forecast_period]['value']))
            data_quantile_0975_cont.append(sum(data_quantile_0975[data_quantile_0975['target'] == forecast_period]['value']))
            data_point_cont.append(sum(data_point[data_point['target'] == forecast_period]['value']))
    
    dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]
    forecast_dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in forecast_dates]
    
    return np.asarray(forecast_dates), np.asarray(dates), np.asarray(data_quantile_0025_cont), np.asarray(data_quantile_0975_cont), np.asarray(data_point_cont), np.asarray([x for x in set(data_quantile['location'].to_numpy())])

ensemble_forecast_directory = '../../data/EuroCOVIDhub-ensemble/'
    
forecast_dates_1wk, dates_1wk, data_quantile_0025_1wk, data_quantile_0975_1wk, data_point_1wk, \
countries = extract_forecasts(ensemble_forecast_directory, '1 wk ahead inc death')

forecast_dates_2wk, dates_2wk, data_quantile_0025_2wk, data_quantile_0975_2wk, data_point_2wk, \
countries = extract_forecasts(ensemble_forecast_directory, '2 wk ahead inc death')

forecast_dates_3wk, dates_3wk, data_quantile_0025_3wk, data_quantile_0975_3wk, data_point_3wk, \
countries = extract_forecasts(ensemble_forecast_directory, '3 wk ahead inc death')

forecast_dates_4wk, dates_4wk, data_quantile_0025_4wk, data_quantile_0975_4wk, data_point_4wk, \
countries = extract_forecasts(ensemble_forecast_directory, '4 wk ahead inc death')

div = 1e3

countries_no_abbr = np.asarray([ISO3166[x] for x in countries])

covid_data = pd.read_csv('../../data/time_series_covid19_deaths_global.csv')

covid_data = covid_data[covid_data["Country/Region"].isin(countries_no_abbr)]

colnames = covid_data.columns.tolist()

start = datetime.datetime.strptime(colnames[4], "%m/%d/%y")
end = datetime.datetime.strptime(colnames[-1], "%m/%d/%y")
dates_covid_data = np.asarray([start + datetime.timedelta(days=x) for x in range(0,(end-start).days+1)])
deaths_covid_data = [covid_data.iloc[:,i].sum() for i in range(4,len(colnames))]
deaths_covid_data = np.array(deaths_covid_data)

deaths_covid_data_diff = np.diff(deaths_covid_data)

def error_comparison(deaths_covid_data, data_point, forecast_dates, dates, dates_covid_data, week):
    error_euler = []
    error_ensemble = []
    
    rate_of_change = []
    euler_prediction_arr = []
    rate_of_change_dates = []
    euler_prediction_dates = []

    for i in range(len(dates)):
        
        ind = np.where(dates_covid_data == dates[i])
            
        if len(ind[0]):
            ind = ind[0][0]
            
            current_rate_of_change = deaths_covid_data[ind]-deaths_covid_data[ind-6]

            ind2 = np.where(dates_covid_data == forecast_dates[i])[0][0]
            euler_prediction = deaths_covid_data[ind2-1]-deaths_covid_data[ind2-1-6]
            print(euler_prediction, dates_covid_data[ind2-1], dates_covid_data[ind2-1-6], dates_covid_data[ind], dates_covid_data[ind-6])
            
            euler_prediction_arr.append(euler_prediction)
            rate_of_change.append(current_rate_of_change)
            rate_of_change_dates.append(forecast_dates[i])
            euler_prediction_dates.append(dates[i])

            error_euler.append(abs(euler_prediction-current_rate_of_change))
            error_ensemble.append(abs(data_point[i]-current_rate_of_change))
      
    error_euler = np.asarray(error_euler)/div  
    error_ensemble = np.asarray(error_ensemble)/div    
    error_euler_cum = np.cumsum(error_euler)
    error_ensemble_cum = np.cumsum(error_ensemble)
    
    rate_of_change = np.array(rate_of_change)
    rate_of_change_dates = np.array(rate_of_change_dates)
    euler_prediction_arr = np.array(euler_prediction_arr)
    euler_prediction_dates = np.array(euler_prediction_dates)

    return rate_of_change, rate_of_change_dates, euler_prediction_arr, euler_prediction_dates, error_euler, error_ensemble, error_euler_cum, error_ensemble_cum

rate_of_change_1wk, rate_of_change_dates_1wk, euler_prediction_arr_1wk, euler_prediction_dates_1wk, error_euler_1wk, error_ensemble_1wk, error_euler_1wk_cum, error_ensemble_1wk_cum = \
error_comparison(deaths_covid_data, data_point_1wk, forecast_dates_1wk, dates_1wk, dates_covid_data, 1)

rate_of_change_2wk, rate_of_change_dates_2wk, euler_prediction_arr_2wk, euler_prediction_dates_2wk, error_euler_2wk, error_ensemble_2wk, error_euler_2wk_cum, error_ensemble_2wk_cum = \
error_comparison(deaths_covid_data, data_point_2wk, forecast_dates_2wk, dates_2wk, dates_covid_data, 2)

rate_of_change_3wk, rate_of_change_dates_3wk, euler_prediction_arr_3wk, euler_prediction_dates_3wk, error_euler_3wk, error_ensemble_3wk, error_euler_3wk_cum, error_ensemble_3wk_cum = \
error_comparison(deaths_covid_data, data_point_3wk, forecast_dates_3wk, dates_3wk, dates_covid_data, 3)

rate_of_change_4wk, rate_of_change_dates_4wk, euler_prediction_arr_4wk, euler_prediction_dates_4wk, error_euler_4wk, error_ensemble_4wk, error_euler_4wk_cum, error_ensemble_4wk_cum = \
error_comparison(deaths_covid_data, data_point_4wk, forecast_dates_4wk, dates_4wk, dates_covid_data, 4)

fig, ax = plt.subplots(ncols = 2, nrows = 2)

#ax[0,0].set_title(r'1 week forecast')
ax[0,0].text(dates_1wk[0]-datetime.timedelta(days=7), 0.9*100, r'(a)')
ax[0,0].plot(rate_of_change_dates_1wk,error_euler_1wk, color = 'tab:red', linewidth = 1.5) 
ax[0,0].plot(rate_of_change_dates_1wk,error_euler_1wk_cum, color = 'tab:red', ls = (0, (5, 1)), linewidth = 1.5) 
ax[0,0].plot(rate_of_change_dates_1wk,error_ensemble_1wk, color = 'tab:blue', linewidth = 1.5)  
ax[0,0].plot(rate_of_change_dates_1wk,error_ensemble_1wk_cum, color = 'tab:blue', ls = (0, (5, 1)), linewidth = 1.5)  
ax[0,0].set_xlim([datetime.date(2021, 5, 1), datetime.date(2021, 12, 1)])
ax[0,0].set_xticks([])
ax[0,0].set_ylabel(r'Prediction error [$\times 10^3$]')
ax[0,0].set_ylim(0,100)

#ax[0,1].set_title(r'2 week forecast')
ax[0,1].text(dates_1wk[0]-datetime.timedelta(days=7), 0.9*100, r'(b)')
ax[0,1].plot(rate_of_change_dates_2wk,error_euler_2wk, color = 'tab:red', linewidth = 1.5) 
ax[0,1].plot(rate_of_change_dates_2wk,error_euler_2wk_cum, color = 'tab:red', ls = (0, (5, 1)), linewidth = 1.5) 
ax[0,1].plot(rate_of_change_dates_2wk,error_ensemble_2wk, color = 'tab:blue', linewidth = 1.5)  
ax[0,1].plot(rate_of_change_dates_2wk,error_ensemble_2wk_cum, color = 'tab:blue', ls = (0, (5, 1)), linewidth = 1.5)  
ax[0,1].set_xlim([datetime.date(2021, 5, 1), datetime.date(2021, 12, 1)])
ax[0,1].set_xticks([])
ax[0,1].set_ylim(0,100)
ax[0,1].set_yticks([])

#ax[1,0].set_title(r'3 week forecast')
ax[1,0].text(dates_1wk[0]-datetime.timedelta(days=7), 0.9*150, r'(c)')
ax[1,0].plot(rate_of_change_dates_3wk,error_euler_3wk, color = 'tab:red', linewidth = 1.5) 
ax[1,0].plot(rate_of_change_dates_3wk,error_euler_3wk_cum, color = 'tab:red', ls = (0, (5, 1)), linewidth = 1.5) 
ax[1,0].plot(rate_of_change_dates_3wk,error_ensemble_3wk, color = 'tab:blue', linewidth = 1.5)  
ax[1,0].plot(rate_of_change_dates_3wk,error_ensemble_3wk_cum, color = 'tab:blue', ls = (0, (5, 1)), linewidth = 1.5)  
ax[1,0].set_xlim([datetime.date(2021, 5, 1), datetime.date(2021, 12, 1)])
ax[1,0].tick_params(axis='x', labelrotation=45)
ax[1,0].set_ylabel(r'Prediction error [$\times 10^3$]')
ax[1,0].set_ylim(0,150)

#ax[1,1].set_title(r'4 week forecast')
ax[1,1].text(dates_1wk[0]-datetime.timedelta(days=7), 0.9*150, r'(d)')
ax[1,1].plot(rate_of_change_dates_4wk,error_euler_4wk, color = 'tab:red', linewidth = 1.5) 
ax[1,1].plot(rate_of_change_dates_4wk,error_euler_4wk_cum, color = 'tab:red', ls = (0, (5, 1)), linewidth = 1.5) 
ax[1,1].plot(rate_of_change_dates_4wk,error_ensemble_4wk, color = 'tab:blue', linewidth = 1.5)  
ax[1,1].plot(rate_of_change_dates_4wk,error_ensemble_4wk_cum, color = 'tab:blue', ls = (0, (5, 1)), linewidth = 1.5)  
ax[1,1].set_xlim([datetime.date(2021, 5, 1), datetime.date(2021, 12, 1)])
ax[1,1].tick_params(axis='x', labelrotation=45)
ax[1,1].set_ylim(0,150)
ax[1,1].set_yticks([])
plt.tight_layout()
plt.savefig('EU_prediction_error.png', dpi = 480)

# plot
fig, ax = plt.subplots(ncols = 2, nrows = 2)

#ax[0,0].set_title(r'1 week forecast')
ax[0,0].text(dates_1wk[0]-datetime.timedelta(days=7), 0.9*30, r'(a)')
ax[0,0].fill_between(dates_1wk, data_quantile_0975_1wk/div, data_quantile_0025_1wk/div, facecolor = 'tab:blue', alpha = 0.5)
ax[0,0].plot(dates_1wk, data_point_1wk/div, color = 'tab:blue', linewidth = 1.5, label = r'CDC ensemble')
ax[0,0].plot(euler_prediction_dates_1wk, euler_prediction_arr_1wk/div, color = 'tab:red', linewidth = 1.5, label = r'Euler')
ax[0,0].plot(rate_of_change_dates_1wk, rate_of_change_1wk/div, ls = (0, (5, 1)), color = 'k', linewidth = 1.5, label = r'reported')
ax[0,0].set_xlim([datetime.date(2021, 5, 1), datetime.date(2021, 12, 1)])
ax[0,0].set_xticks([])
ax[0,0].set_ylabel(r'Weekly deaths [$\times 10^3$]')
ax[0,0].set_ylim(0,30)

#ax[0,1].set_title(r'2 week forecast')
ax[0,1].text(dates_1wk[0]-datetime.timedelta(days=7), 0.9*30, r'(b)')
ax[0,1].fill_between(dates_2wk, data_quantile_0975_2wk/div, data_quantile_0025_2wk/div, facecolor = 'tab:blue', alpha = 0.5)
ax[0,1].plot(dates_2wk, data_point_2wk/div, color = 'tab:blue', linewidth = 1.5)
ax[0,1].plot(euler_prediction_dates_2wk, euler_prediction_arr_2wk/div, color = 'tab:red', linewidth = 1.5)
ax[0,1].plot(rate_of_change_dates_2wk, rate_of_change_2wk/div, ls = (0, (5, 1)), color = 'k', linewidth = 1.5)
ax[0,1].set_xlim([datetime.date(2021, 5, 1), datetime.date(2021, 12, 1)])
ax[0,1].set_xticks([])
ax[0,1].set_ylim(0,30)
ax[0,1].set_yticks([])

#ax[1,0].set_title(r'3 week forecast')
ax[1,0].text(dates_1wk[0]-datetime.timedelta(days=7), 0.9*30, r'(c)')
ax[1,0].fill_between(dates_3wk, data_quantile_0975_3wk/div, data_quantile_0025_3wk/div, facecolor = 'tab:blue', alpha = 0.5)
ax[1,0].plot(dates_3wk, data_point_3wk/div, color = 'tab:blue', linewidth = 1.5)
ax[1,0].plot(euler_prediction_dates_3wk, euler_prediction_arr_3wk/div, color = 'tab:red', linewidth = 1.5)
ax[1,0].plot(rate_of_change_dates_3wk, rate_of_change_3wk/div, ls = (0, (5, 1)), color = 'k', linewidth = 1.5)
ax[1,0].set_xlim([datetime.date(2021, 5, 1), datetime.date(2021, 12, 1)])
ax[1,0].tick_params(axis='x', labelrotation=45)
ax[1,0].set_ylabel(r'Weekly deaths [$\times 10^3$]')
ax[1,0].set_ylim(0,30)

#ax[1,1].set_title(r'4 week forecast')
ax[1,1].text(dates_1wk[0]-datetime.timedelta(days=7), 0.9*30, r'(d)')
ax[1,1].fill_between(dates_4wk, data_quantile_0975_4wk/div, data_quantile_0025_4wk/div, facecolor = 'tab:blue', alpha = 0.5)
ax[1,1].plot(dates_4wk, data_point_4wk/div, color = 'tab:blue', linewidth = 1.5)
ax[1,1].plot(euler_prediction_dates_4wk, euler_prediction_arr_4wk/div, color = 'tab:red', linewidth = 1.5)
ax[1,1].plot(rate_of_change_dates_4wk, rate_of_change_4wk/div, ls = (0, (5, 1)), color = 'k', linewidth = 1.5)
ax[1,1].set_xlim([datetime.date(2021, 5, 1), datetime.date(2021, 12, 1)])
ax[1,1].tick_params(axis='x', labelrotation=45)
ax[1,1].set_ylim(0,30)
ax[1,1].set_yticks([])
 
plt.tight_layout()
plt.savefig("EU_ensemble_forecast.png", dpi = 480)