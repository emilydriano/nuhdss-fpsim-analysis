"""
Standard plots used in calibrations
"""
import fpsim as fp
import sciris as sc
import pylab as pl
import seaborn as sns
import pandas as pd
import numpy as np


# Settings
min_age = 15
max_age = 50
bin_size = 5
do_save = True

# TODO: Pull this from fpsim rather than hardcoding
age_bin_map = {
    '10-14': [10, 15],
    '15-19': [15, 20],
    '20-24': [20, 25],
    '25-29': [25, 30],
    '30-34': [30, 35],
    '35-39': [35, 40],
    '40-44': [40, 45],
    '45-49': [45, 50]
}


def plot_by_age(sim, do_save=do_save):

    fig, ax = pl.subplots()
    age_bins = [18, 20, 25, 35, 50]
    colors = sc.vectocolor(age_bins)
    cind = 0

    for alabel, ares in sim.get_analyzer('cpr_by_age').results.items():
        ax.plot(sim.results.t, ares, label=alabel, color=colors[cind])
        cind += 1
    ax.legend(loc='best', frameon=False)
    ax.set_ylim([0, 1])
    ax.set_ylabel('CPR')
    ax.set_title('CPR')
    pl.show()
    if do_save: sc.savefig('figures/cpr_by_age.png')

    fig, ax = pl.subplots()
    df = pd.DataFrame(sim.get_analyzer('method_mix_by_age').results)
    df['method'] = sim.contraception_module.methods.keys()
    df_plot = df.melt(id_vars='method')
    sns.barplot(x=df_plot['method'], y=df_plot['value'], ax=ax, hue=df_plot['variable'], palette="viridis")
    pl.show()
    if do_save: sc.savefig('figures/method_mix_by_age.png')

    return


def plot_asfr(data, res, pars, do_save=do_save):
    # Print ASFR form model in output
    for key in fp.age_bin_map.keys():
        print(f'ASFR (annual) for age bin {key} in the last year of the sim: {res["asfr"][key][-1]}')

    x = [1, 2, 3, 4, 5, 6, 7, 8]

    # Load data
    year = data[data['year'] == pars['end_year']]
    asfr_data = year.drop(['year'], axis=1).values.tolist()[0]

    x_labels = []
    asfr_model = []

    # Extract from model
    for key in fp.age_bin_map.keys():
            x_labels.append(key)
            asfr_model.append(res['asfr'][key][-1])

    # Plot
    fig, ax = pl.subplots()
    kw = dict(lw=3, alpha=0.7, markersize=10)
    ax.plot(x, asfr_data, marker='^', color='black', label="UN data", **kw)
    ax.plot(x, asfr_model, marker='*', color='cornflowerblue', label="FPsim", **kw)
    pl.xticks(x, x_labels)
    pl.ylim(bottom=-10)
    ax.set_title(f'Age specific fertility rate per 1000 woman years')
    ax.set_xlabel('Age')
    ax.set_ylabel('ASFR in 2019')
    ax.legend(frameon=False)
    sc.boxoff()

    pl.show()
    if do_save: sc.savefig('figures/asfr.png')
    return


def plot_methods(data_methods, data_use, sim, do_save=do_save):
    """
    Plots both dichotomous method data_use and non-data_use and contraceptive mix
    """
    ppl = sim.people

    # Setup
    model_labels_all = [m.label for m in sim.contraception_module.methods.values()]
    model_labels_methods = sc.dcp(model_labels_all)
    model_method_counts = sc.odict().make(keys=model_labels_all, vals=0.0)

    # Extract from model
    # TODO: refactor, this shouldn't need to loop over people, can just data_use a histogram
    for i in range(len(ppl)):
        if ppl.alive[i] and not ppl.sex[i] and ppl.age[i] >= min_age and ppl.age[i] < max_age:
                model_method_counts[ppl.method[i]] += 1

    model_method_counts[:] /= model_method_counts[:].sum()

    # Method mix from data - country PMA data (mix.csv)
    data_methods_mix = {
            'Pill': data_methods.loc[data_methods['method'] == 'Pill', 'perc'].iloc[0],
            'IUDs': data_methods.loc[data_methods['method'] == 'IUDs', 'perc'].iloc[0],
            'Injectables': data_methods.loc[data_methods['method'] == 'Injectables', 'perc'].iloc[0],
            'Condoms': data_methods.loc[data_methods['method'] == 'Condoms', 'perc'].iloc[0],
            'BTL': data_methods.loc[data_methods['method'] == 'BTL', 'perc'].iloc[0],
            'Withdrawal': data_methods.loc[data_methods['method'] == 'Withdrawal', 'perc'].iloc[0],
            'Implants': data_methods.loc[data_methods['method'] == 'Implants', 'perc'].iloc[0],
            'Other traditional': data_methods.loc[data_methods['method'] == 'Other traditional', 'perc'].iloc[0],
            'Other modern': data_methods.loc[data_methods['method'] == 'Other modern', 'perc'].iloc[0]
    }

    # Method data_use from data - country PMA data (data_use.csv)
    no_use = data_use.loc[0, 'perc']
    any_method = data_use.loc[1, 'perc']
    data_methods_use = {
            'No data_use': no_use,
            'Any method': any_method
    }

    # Plot bar charts of method mix and data_use among users

    # Calculate users vs non-users in model
    model_methods_mix = sc.dcp(model_method_counts)
    model_use = [model_methods_mix['None'], model_methods_mix[1:].sum()]
    model_use_percent = [i * 100 for i in model_use]

    # Calculate mix within users in model
    model_methods_mix['None'] = 0.0
    model_users_sum = model_methods_mix[:].sum()
    model_methods_mix[:] /= model_users_sum
    mix_model = model_methods_mix.values()[1:]
    mix_percent_model = [i * 100 for i in mix_model]

    # Set method data_use and mix from data
    mix_percent_data = list(data_methods_mix.values())
    data_use_percent = list(data_methods_use.values())

    # Set up plotting
    use_labels = list(data_methods_use.keys())
    df_mix = pd.DataFrame({'PMA': mix_percent_data, 'FPsim': mix_percent_model}, index=model_labels_methods[1:])
    df_use = pd.DataFrame({'PMA': data_use_percent, 'FPsim': model_use_percent}, index=use_labels)

    # Plot mix
    ax = df_mix.plot.barh(color={'PMA':'black', 'FPsim':'cornflowerblue'})
    ax.set_xlabel('Percent users')
    ax.set_title(f'Contraceptive Method Mix - Model vs Data')
    if do_save:
        sc.savefig("figures/method_mix.png")

    # Plot data_use
    ax = df_use.plot.barh(color={'PMA':'black', 'FPsim':'cornflowerblue'})
    ax.set_xlabel('Percent')
    ax.set_title(f'Contraceptive Method Use - Model vs Data')
    if do_save:
        sc.savefig("figures/method_use.png")

    pl.show()


def plot_parity(ppl, ageparity_data, do_save=do_save, ageparity_dataset='PMA 2022'):
    '''
    Plot an age-parity distribution for model vs data
    '''
    # Set up
    age_keys = list(fp.age_bin_map.keys())[1:]
    age_bins = pl.arange(min_age, max_age, bin_size)
    parity_bins = pl.arange(0, 7)  # Plot up to parity 6
    n_age = len(age_bins)
    n_parity = len(parity_bins)

    # Load data
    ageparity_data = ageparity_data[ageparity_data['dataset'] == ageparity_dataset]
    sky_arr = sc.odict()

    sky_arr['Data'] = pl.zeros((len(age_keys), len(parity_bins)))

    for age, row in ageparity_data.iterrows():
        if row.age in age_keys and row.parity <7:
                age_ind = age_keys.index(row.age)
                sky_arr['Data'][age_ind, row.parity] = row.percentage

    # Extract from model
    # TODO, refactor - can just use histogram instead of looping over agents
    sky_arr['Model'] = pl.zeros((len(age_bins), len(parity_bins)))
    for i in range(len(ppl)):
        if ppl.alive[i] and not ppl.sex[i] and ppl.age[i] >= min_age and ppl.age[i] < max_age:
                age_bin = sc.findinds(age_bins <= ppl.age[i])[-1]
                parity_bin = sc.findinds(parity_bins <= ppl.parity[i])[-1]
                sky_arr['Model'][age_bin, parity_bin] += 1

    # Normalize
    for key in ['Data', 'Model']:
            sky_arr[key] /= sky_arr[key].sum() / 100

    # Find diff to help visualize in plotting
    sky_arr['Diff_data-model'] = sky_arr['Data']-sky_arr['Model']

    # Plot ageparity
    for key in ['Data', 'Model', 'Diff_data-model']:
        fig = pl.figure(figsize=(20, 14))

        pl.pcolormesh(sky_arr[key], cmap='parula')
        pl.xlabel('Age', fontweight='bold')
        pl.ylabel('Parity', fontweight='bold')
        pl.title(f'Age-parity plot for the {key.lower()}\n\n', fontweight='bold')
        pl.gca().set_xticks(pl.arange(n_age))
        pl.gca().set_yticks(pl.arange(n_parity))
        pl.gca().set_xticklabels(age_bins)
        pl.gca().set_yticklabels(parity_bins)
        #pl.gca().view_init(30, 45)
        pl.draw()

        if do_save:
            sc.savefig(f"figures/ageparity_{key.lower()}.png")

        pl.show()


def plot_cpr(data_cpr, res, pars, do_save=do_save):
    '''
    Plot contraceptive prevalence rate for model vs data
    '''
    # Import data
    data_cpr = data_cpr[data_cpr['year'] <= pars['end_year']] # Restrict years to plot

    # Plot
    pl.plot(data_cpr['year'], data_cpr['cpr'], label='UN Data Portal', color='black')
    pl.plot(res['t'], res['cpr']*100, label='FPsim', color='cornflowerblue')
    pl.xlabel('Year')
    pl.ylabel('Percent')
    pl.title(f'Contraceptive Prevalence Rate - Model vs Data')
    pl.legend()

    if do_save:
        sc.savefig("figures/cpr.png")

    pl.show()


def plot_contra_analysis(res, pars, do_save=do_save):
    years = np.arange(pars['start_year'], pars['end_year']+1)

    # Plot
    pl.plot(years, res['contra_access_by_year'], label='Contra Access', color='black')
    pl.plot(years, res['new_users_by_year'], label='New Users', color='cornflowerblue')
    pl.xlabel('Year')
    pl.ylabel('Number of Agents')
    pl.title(f'Contra Access vs New Users')
    pl.legend()

    if do_save:
        sc.savefig("figures/contra_analysis.png")

    pl.show()


def plot_tfr(data_tfr, res, do_save=do_save):
    '''
    Plot total fertility rate for model vs data
    '''

    # Plot
    pl.plot(data_tfr['year'], data_tfr['tfr'], label='World Bank', color='black')
    pl.plot(res['tfr_years'], res['tfr_rates'], label='FPsim', color='cornflowerblue')
    pl.xlabel('Year')
    pl.ylabel('Rate')
    pl.title(f'Total Fertility Rate - Model vs Data')
    pl.legend()

    if do_save:
        sc.savefig("figures/tfr.png")

    pl.show()


def pop_growth_rate(years, population):
    '''
    Calculates growth rate as a time series to help compare model to data
    '''
    growth_rate = np.zeros(len(years) - 1)

    for i in range(len(years)):
            if population[i] == population[-1]:
                    break
            growth_rate[i] = ((population[i + 1] - population[i]) / population[i]) * 100

    return growth_rate


def plot_pop_growth(data_popsize, res, pars, do_save=do_save):
    '''
    Plot annual population growth rate for model vs data
    '''
    # Import data
    data_popsize = data_popsize[data_popsize['year'] <= pars['end_year']]  # Restrict years to plot

    data_pop_years = data_popsize['year'].to_numpy()
    data_population = data_popsize['population'].to_numpy()

    # Extract from model
    model_growth_rate = pop_growth_rate(res['tfr_years'], res['pop_size'])

    data_growth_rate = pop_growth_rate(data_pop_years, data_population)

    # Plot
    pl.plot(data_pop_years[1:], data_growth_rate, label='World Bank', color='black')
    pl.plot(res['tfr_years'][1:], model_growth_rate, label='FPsim', color='cornflowerblue')
    pl.xlabel('Year')
    pl.ylabel('Rate')
    pl.title(f'Population Growth Rate - Model vs Data')
    pl.legend()

    if do_save:
        sc.savefig("figures/popgrowth.png")

    pl.show()


def plot_birth_space_afb(data_spaces, data_afb, ppl, do_save=do_save):
    '''
    Plot birth space and age at first birth for model vs data
    '''
    # Set up
    spacing_bins = sc.odict({'0-12': 0, '12-24': 1, '24-48': 2, '>48': 4})  # Spacing bins in months
    model_age_first = []
    model_spacing = []
    model_spacing_counts = sc.odict().make(keys=spacing_bins.keys(), vals=0.0)
    data_spacing_counts = sc.odict().make(keys=spacing_bins.keys(), vals=0.0)

    # Extract age at first birth and birth spaces from model
    # TODO, refactor to avoid loops
    for i in range(len(ppl)):
        if ppl.alive[i] and not ppl.sex[i] and min_age <= ppl.age[i] < max_age:
                if ppl.first_birth_age[i] == -1:
                        model_age_first.append(float('inf'))
                else:
                        model_age_first.append(ppl.first_birth_age[i])
                        if ppl.parity[i] > 1:
                                cleaned_birth_ages = ppl.birth_ages[i][~np.isnan(ppl.birth_ages[i])]
                                for d in range(len(cleaned_birth_ages) - 1):
                                        space = cleaned_birth_ages[d + 1] - cleaned_birth_ages[d]
                                        if space > 0:
                                                ind = sc.findinds(space > spacing_bins[:])[-1]
                                                model_spacing_counts[ind] += 1
                                                model_spacing.append(space)

    # Normalize model birth space bin counts to percentages
    model_spacing_counts[:] /= model_spacing_counts[:].sum()
    model_spacing_counts[:] *= 100

    age_first_birth_model = pd.DataFrame(data=model_age_first)

    # Extract birth spaces and age at first birth from data
    for i, j in data_spaces.iterrows():
        space = j['space_mo'] / 12
        ind = sc.findinds(space > spacing_bins[:])[-1]
        data_spacing_counts[ind] += j['Freq']

    age_first_birth_data = pd.DataFrame(data=data_afb)

    # Normalize dat birth space bin counts to percentages
    data_spacing_counts[:] /= data_spacing_counts[:].sum()
    data_spacing_counts[:] *= 100

    # Plot age at first birth (histogram with KDE)
    sns.histplot(data=age_first_birth_model, stat='proportion', kde=True, binwidth=1, color='cornflowerblue', label='FPsim')
    sns.histplot(x=age_first_birth_data['afb'], stat='proportion', kde=True, weights=age_first_birth_data['wt'], binwidth=1, color='dimgrey', label='DHS data')
    pl.xlabel('Age at first birth')
    pl.title(f'Age at First Birth - Model vs Data')
    pl.legend()

    if do_save:
        sc.savefig("figures/age_first_birth.png")

    pl.show()

    # Plot birth space bins with diff
    data_spacing_bins = np.array(data_spacing_counts.values())
    model_spacing_bins = np.array(model_spacing_counts.values())

    diff = model_spacing_bins - data_spacing_bins

    res_bins = np.array([[model_spacing_bins], [data_spacing_bins], [diff]])

    bins_frame = pd.DataFrame(
            {'Model': model_spacing_bins, 'Data': data_spacing_bins, 'Diff': diff},
            index=spacing_bins.keys())

    print(bins_frame) # Print in output, remove if not needed

    ax = bins_frame.plot.barh(color={'Data': 'black', 'Model': 'cornflowerblue', 'Diff': 'red'})
    ax.set_xlabel('Percent of live birth spaces')
    ax.set_ylabel('Birth space in months')
    ax.set_title(f'Birth Space Bins - Model vs Data')

    if do_save:
        sc.savefig("figures/birth_space_bins.png")

    pl.show()


def plot_ever_used(sim):
    '''
    Plot percentage of women who have ever used contraception over time
    '''
    res = sim.results
    # Plot
    pl.plot(res['t'], res['ever_used_contra'], color='cornflowerblue')
    pl.xlabel('Year')
    pl.ylabel('Perc of Women')
    pl.title(f'Perc of Women Ever Used Contra')

    if do_save:
        sc.savefig("figures/ever_used.png")

    pl.show()

    return


def plot_urban(sim):
    '''
    Plot percentage of women in urban areas over time
    '''
    res = sim.results
    # Plot
    pl.plot(res['t'], res['urban_women'], color='cornflowerblue')
    pl.xlabel('Year')
    pl.ylabel('Perc of Women')
    pl.title(f'Perc of Women in Urban Areas')

    if do_save:
        sc.savefig("figures/urban.png")

    pl.show()
    
    return


def plot_parity(sim):
    '''
    Plot parity over time
    '''
    res = sim.results

    # Plotting the lines
    pl.plot(res['t'], res['parity0to1'], label='Parity 0-1', color='blue')
    pl.plot(res['t'], res['parity2to3'], label='Parity 2-3', color='red')
    pl.plot(res['t'], res['parity4to5'], label='Parity 4-5', color='green')
    pl.plot(res['t'], res['parity6plus'], label='Parity 6+', color='orange')

    # Adding title and labels
    pl.title('Parity over Time')
    pl.xlabel('Time')
    pl.ylabel('Percentage of Women')
    pl.legend()

    if do_save:
        sc.savefig("figures/parity.png")

    pl.show()

    return


def plot_wealthquintiles(sim):
    res = sim.results

    # Plotting the lines
    pl.plot(res['t'], res['wq1'], label='WQ1', color='blue')
    pl.plot(res['t'], res['wq2'], label='WQ2', color='red')
    pl.plot(res['t'], res['wq3'], label='WQ3', color='green')
    pl.plot(res['t'], res['wq4'], label='WQ4', color='orange')
    pl.plot(res['t'], res['wq5'], label='WQ5', color='purple')

    # Adding title and labels
    pl.title('Distribution of Wealth Quintiles over Time')
    pl.xlabel('Time')
    pl.ylabel('Percentage of Women')
    pl.legend()

    if do_save:
        sc.savefig("figures/wealthquintiles.png")

    pl.show()

    return


def plot_paid_work(data_employment, sim, do_save=do_save):
    '''
    Plot rates of paid employment between model and data
    '''
    # Extract paid work from data
    data_empowerment = data_employment.iloc[1:-1]
    data_paid_work = data_empowerment[['age', 'paid_employment', 'paid_employment.se']].copy()
    age_bins = np.arange(min_age, max_age + 1, bin_size)
    data_paid_work['age_group'] = pd.cut(data_paid_work['age'], bins=age_bins, right=False)

    # Calculate mean and standard error for each age bin
    employment_data_grouped = data_paid_work.groupby('age_group', observed=False)['paid_employment']
    employment_data_mean = employment_data_grouped.mean().tolist()
    employment_data_se = data_paid_work.groupby('age_group', observed=False)['paid_employment.se'].apply(
        lambda x: np.sqrt(np.sum(x ** 2)) / len(x)).tolist()

    # Extract paid work from model
    employed_counts = {age_bin: 0 for age_bin in age_bins}
    total_counts = {age_bin: 0 for age_bin in age_bins}

    # Count the number of employed and total people in each age bin
    ppl = sim.people
    for i in range(len(ppl)):
        if ppl.alive[i] and not ppl.sex[i] and min_age <= ppl.age[i] < max_age:
            age_bin = age_bins[sc.findinds(age_bins <= ppl.age[i])[-1]]
            total_counts[age_bin] += 1
            if ppl.paid_employment[i]:
                employed_counts[age_bin] += 1

    # Calculate the percentage of employed people in each age bin and their standard errors
    percentage_employed = {}
    percentage_employed_se = {}
    age_bins = np.arange(min_age, max_age, bin_size)
    for age_bin in age_bins:
        total_ppl = total_counts[age_bin]
        if total_ppl != 0:
            employed_ratio = employed_counts[age_bin] / total_ppl
            percentage_employed[age_bin] = employed_ratio
            percentage_employed_se[age_bin] = (employed_ratio * (
                    1 - employed_ratio) / total_ppl) ** 0.5
        else:
            percentage_employed[age_bin] = 0
            percentage_employed_se[age_bin] = 0

    employment_model = list(percentage_employed.values())
    employment_model_se = list(percentage_employed_se.values())

    # Set up plotting
    labels = list(age_bin_map.keys())[1:]
    x_pos = np.arange(len(labels))
    fig, ax = pl.subplots()
    width = 0.35

    # Plot Data
    ax.barh(x_pos - width / 2, employment_data_mean, width, label='DHS', color='black')
    ax.errorbar(employment_data_mean, x_pos - width / 2, xerr=employment_data_se, fmt='none', ecolor='gray',
                capsize=5)

    # Plot Model
    ax.barh(x_pos + width / 2, employment_model, width, label='FPsim', color='cornflowerblue')
    ax.errorbar(employment_model, x_pos + width / 2, xerr=employment_model_se, fmt='none', ecolor='gray',
                capsize=5)

    # Set labels and title
    ax.set_xlabel('Percent Women with Paid Work')
    ax.set_ylabel('Age Bin')
    ax.set_title(f'Kenya: Paid Employment - Model vs Data')
    ax.set_yticks(x_pos)
    ax.set_yticklabels(labels)
    ax.legend()

    if do_save:
        pl.savefig(f"figures/paid_employment.png", bbox_inches='tight', dpi=100)
    pl.show()

def plot_education(data_education, sim, do_save=do_save):
    '''
    Plot years of educational attainment between model and data
    '''
    pl.clf()

    # Extract education from data
    data_edu = data_education[['age', 'edu', 'se']].sort_values(by='age')
    data_edu = data_edu.query(f"{min_age} <= age < {max_age}").copy()
    age_bins = np.arange(min_age, max_age + 1, bin_size)
    data_edu['age_group'] = pd.cut(data_edu['age'], bins=age_bins, right=False)

    # Calculate mean and standard error for each age bin
    education_data_grouped = data_edu.groupby('age_group', observed=False)['edu']
    education_data_mean = education_data_grouped.mean().tolist()
    education_data_se = data_edu.groupby('age_group', observed=False)['se'].apply(
        lambda x: np.sqrt(np.sum(x ** 2)) / len(x)).tolist()

    # Extract education from model
    model_edu_years = {age_bin: [] for age_bin in np.arange(min_age, max_age, bin_size)}
    ppl = sim.people
    for i in range(len(ppl)):
        if ppl.alive[i] and not ppl.sex[i] and min_age <= ppl.age[i] < max_age:
            age_bin = age_bins[sc.findinds(age_bins <= ppl.age[i])[-1]]
            model_edu_years[age_bin].append(ppl.edu_attainment[i])

    # Calculate average # of years of educational attainment for each age
    model_edu_mean = []
    model_edu_se = []
    for age_group in model_edu_years:
        if len(model_edu_years[age_group]) != 0:
            avg_edu = sum(model_edu_years[age_group]) / len(model_edu_years[age_group])
            se_edu = np.std(model_edu_years[age_group], ddof=1) / np.sqrt(len(model_edu_years[age_group]))
            model_edu_mean.append(avg_edu)
            model_edu_se.append(se_edu)
        else:
            model_edu_years[age_group] = 0
            model_edu_se.append(0)

    # Set up plotting
    labels = list(age_bin_map.keys())[1:]
    x_pos = np.arange(len(labels))
    fig, ax = pl.subplots()
    width = 0.35

    # Plot DHS data
    ax.barh(x_pos - width / 2, education_data_mean, width, label='DHS', color='black')
    ax.errorbar(education_data_mean, x_pos - width / 2, xerr=education_data_se, fmt='none', ecolor='gray',
                capsize=5)

    # Plot FPsim data
    ax.barh(x_pos + width / 2, model_edu_mean, width, label='FPsim', color='cornflowerblue')
    ax.errorbar(model_edu_mean, x_pos + width / 2, xerr=model_edu_se, fmt='none', ecolor='gray', capsize=5)

    # Set labels and title
    ax.set_xlabel('Avg Years of Education Attainment')
    ax.set_ylabel('Age Bin')
    ax.set_title(f'Kenya: Years of Education - Model vs Data')
    ax.set_yticks(x_pos)
    ax.set_yticklabels(labels)
    ax.legend()

    if do_save:
        pl.savefig(f"figures/education.png", bbox_inches='tight', dpi=100)
    pl.show()


def plot_all(sim, val_data):
    # Make all the above plots
    plot_by_age(sim)
    plot_asfr(val_data['asfr'], sim.results, sim.pars)
    plot_methods(val_data['mix'], val_data['use'], sim)
    plot_parity(sim.people, val_data['ageparity'])
    plot_cpr(val_data['cpr'], sim.results, sim.pars)
    plot_contra_analysis(sim.results, sim.pars)
    plot_tfr(val_data['tfr'], sim.results)
    plot_pop_growth(val_data['popsize'], sim.results, sim.pars)
    plot_birth_space_afb(val_data['birth_spacing_dhs'], val_data['afb.table'], sim.people)
    return


def plot_calib(sim, val_data):
    plot_methods(val_data['mix'], val_data['use'], sim)
    plot_cpr(val_data['cpr'], sim.results, sim.pars)
    plot_tfr(val_data['tfr'], sim.results)
    plot_birth_space_afb(val_data['birth_spacing_dhs'], val_data['afb.table'], sim.people)
    plot_asfr(val_data['asfr'], sim.results, sim.pars)
    return
