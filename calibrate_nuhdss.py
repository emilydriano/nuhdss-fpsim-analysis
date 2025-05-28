import os
from locations import nuhdss
from plotting import *

# Register custom location; this is necessary to include when setting up any scripts using the 'nuhdss' location
fp.defaults.register_location('nuhdss', nuhdss)

def run_sim():
    pars = fp.pars(location='nuhdss')

    # Settings
    pars['n_agents'] = 10_000
    pars['start_year'] = 2000
    pars['end_year'] = 2020

    # Free parameters for calibration
    pars['fecundity_var_low'] = .8
    pars['fecundity_var_high'] = 2
    pars['exposure_factor'] = 1.25

    # Adjust contraceptive choice parameters
    cm_pars = dict(
        prob_use_year=2020,
        prob_use_trend_par=.04,
        method_weights=np.array([0.38, .8, 0.6, 0.8, 0.8, 1, 1.6, 0.7, 8]),
    )

    method_choice = fp.SimpleChoice(pars=cm_pars, location='nuhdss')
    sim = fp.Sim(
        pars=pars,
        contraception_module=method_choice,
        analyzers=[fp.cpr_by_age(), fp.method_mix_by_age(), fp.education_recorder()],
        education_module=fp.Education(location='nuhdss')
    )
    sim.run()

    return sim


if __name__ == '__main__':

    sim = run_sim()

    # Path to local nuhdss/data directory
    data_dir = './locations/nuhdss/data'

    # Load in validation data
    val_data_list = ['ageparity', 'use', 'birth_spacing_dhs', 'afb.table', 'cpr', 'asfr', 'mix', 'tfr', 'popsize', 'education']
    val_data = sc.objdict()
    for vd in val_data_list:
        file_path = os.path.join(data_dir, f"{vd}.csv")
        val_data[vd] = pd.read_csv(file_path)

    # Plot calibration figures
    plot_calib(sim, val_data)

    # Plot covariate figures
    edu_analyzer = sim.get_analyzer('education_recorder')
    edu_analyzer.plot_waterfall()