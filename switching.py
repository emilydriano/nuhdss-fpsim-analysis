import fpsim as fp
import sciris as sc
import pandas as pd
import numpy as np
from locations import nuhdss

# Register custom location; this is necessary to include when setting up any scripts using the 'nuhdss' location
fp.defaults.register_location('nuhdss', nuhdss)

class StoreMethods(fp.Analyzer):
    def __init__(self, age_groups):
        super().__init__()
        self.age_groups = age_groups
        self.data = {age_group: [] for age_group in age_groups}
    
    def apply(self, sim):
        for age_group in self.age_groups:
            methods = sc.dcp(sim['methods']['adjusted'])
            self.data[age_group].append(methods)

def generate_random_matrix(size):
    """Generate a random switching matrix."""
    matrix = np.random.rand(size, size)
    matrix /= matrix.sum(axis=1)[:, np.newaxis]  # Normalize rows to sum to 1
    return matrix.tolist()

def run_simulation(coverage=0.6):
    # Define simulation parameters
    n_agents = 100000
    start_year = 2012
    end_year = 2030
    
    # Define age groups
    age_groups = ['<20', '20-24', '25-34', '35+'] #'25-29' '30-34',

    # Generate switching matrices for each age group
    switching_matrices = {age_group: generate_random_matrix(len(age_groups)) for age_group in age_groups}

    # Initialize simulation with the analyzer
    analyzer = StoreMethods(age_groups)
    
    # Define effect size for the intervention
    effect_size = 0.5
    init_factor = 1.0 + effect_size * coverage
    
    # Define scenarios with the specified coverage
    scenarios = []
    s1 = fp.make_scen(method='Injectables', init_factor=init_factor, year=2024)
    s2 = fp.make_scen(method='Pill', init_factor=init_factor, year=2024)
    s3 = fp.make_scen(method='Withdrawal', init_factor=init_factor, year=2024)
    s4 = fp.make_scen(method='Condoms', init_factor=init_factor, year=2024)
    s5 = fp.make_scen(method='Implants', init_factor=init_factor, year=2024)
    s6 = fp.make_scen(method='IUDs', init_factor=init_factor, year=2024)
    s7 = fp.make_scen(method='BTL', init_factor=init_factor, year=2024)
    s8 = fp.make_scen(method='Other modern', init_factor=init_factor, year=2024)
    s9 = fp.make_scen(method='Other traditional', init_factor=init_factor, year=2024)
    s10 = s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8 + s9
    scenarios.append((s10, f'Campaign with { coverage*100:.1f}% coverage'))

    # Set the parameters for the simulation including location, number of agents, and time frame
    pars = fp.pars(location='nuhdss', n_agents=n_agents, start_year=start_year, end_year=end_year)

    # Create a Scenarios object with the defined parameters and set the number of repeats
    scens = fp.Scenarios(pars=pars, repeats=3)

    # Add the baseline scenario
    scens.add_scen(label='Baseline')

    # Add the campaign scenario with 60% coverage
    for scen, label in scenarios:
        scens.add_scen(scen, label=label)

    # Run the simulation for all scenarios
    scens.run()
    
    # Plot the results of the simulation and capture the figure
    scens.plot()
    
    # Access and save the data for each age group
    for age_group in age_groups:
        if analyzer.data[age_group]:
            methods_data = analyzer.data[age_group][-1]  # Get the last step data for the age group
            
            # Convert the methods data to a pandas DataFrame
            df = pd.DataFrame(methods_data)
            
            # Replace invalid characters in age_group for filename
            safe_age_group = age_group.replace('<', 'under_').replace('+', 'plus').replace('-', '_')

            # Save the DataFrame to a CSV file
            df.to_csv(f'methods_data_{safe_age_group}.csv', index=False)
            print(f"Data saved to methods_data_{safe_age_group}.csv")

# Run the simulation function with 60% coverage
run_simulation(coverage=0.6)