"""
Microgrid Energy Management System (EMS) Simulation
"""
# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pyomo.environ as pyomo
from pyomo.opt import SolverFactory
import sys
# --------------------------
# Configuration Parameters
# --------------------------
SOLAR_CAPACITY = 2500  # kW (2.5 MW)
DIESEL_CAPACITY = 1500  # kW (1.5 MW)
BATTERY_CAPACITY = 5000  # kWh (5 MWh)
BATTERY_MAX_RATE = 2500  # kW (2.5 MW charge/discharge rate from specs)
DIESEL_COST = 0.25  # $/kWh
BATTERY_COST = 0.05  # $/kWh throughput
DIESEL_EMISSION_FACTOR = 0.8  # kg CO2/kWh
GRID_EXPORT_LIMIT = 3000  # kW (adjust based on your grid connection capacity)

# --------------------------
# Data Loading & Preprocessing
# --------------------------
def load_and_preprocess_data(solar_file, load_file, price_file):
    """
    Load and align solar & load data with error handling
    Returns DataFrame with columns: timestamp, solar, load
    """
    try:
        # Load data with explicit column names
        solar_df = pd.read_csv(
            solar_file,
            parse_dates=['Time'],
            index_col='Time',
            usecols=['Time', 'Production'],
            dtype={'Production': float}
        ).rename(columns={'Production': 'solar'})

        load_df = pd.read_csv(
            load_file,
            parse_dates=['TIMESTAMP'],
            index_col='TIMESTAMP',
            usecols=['TIMESTAMP', 'VALUE'],
            dtype={'VALUE': float}
        ).rename(columns={'VALUE': 'load'})

        price_df = pd.read_csv(
        price_file,  
        parse_dates=['TIMESTAMP'],
        index_col='TIMESTAMP',
        usecols=['TIMESTAMP', 'PRICE'],  # Keep actual column names
        dtype={'PRICE': float}
    ).rename(columns={'PRICE': 'market_price'})
        
    except Exception as e:
    
        print(f"Data loading failed: {str(e)}")
        sys.exit(1)


      # Convert MW to kW
    solar_df['solar'] *= 1000
    load_df['load'] *= 1000
    price_df['market_price'] /= 1000

        # Merge data using inner join to ensure alignment
    df = pd.merge(
            solar_df,
            load_df,
            left_index=True,
            right_index=True,
            how='inner'
        )

        # Add price data to merge
    df = pd.merge(
        df,
        price_df,
        left_index=True,
        right_index=True,
        how='inner'
    )

    # Validate data
    if df.isnull().sum().sum() > 0:
            df = df.ffill().bfill()
            print("Warning: Missing values filled using forward/backward fill")
    print(len(df))
    if len(df) != 898:  # 15-min intervals * 24 hrs * 10 days
            raise ValueError("Data doesn't contain exactly 10 days of 15-min data")

    return df.reset_index().rename(columns={'index': 'timestamp', 'Time': 'timestamp'})
# --------------------------
# Pyomo Optimization Model
# --------------------------
def build_optimization_model(data, diesel_enabled=True):
    """
    Build Pyomo optimization model with configurable diesel generator
    """
    model = pyomo.ConcreteModel()
    
    # Time periods (15-min intervals)
    model.T = pyomo.Set(initialize=data.index)

    # Add these lines after model.T definition
    model.solar_data = pyomo.Param(model.T, initialize=data['solar'].to_dict())
    model.load_data = pyomo.Param(model.T, initialize=data['load'].to_dict())
    # --------------------------
    # Decision Variables
    # --------------------------
    # Grid interaction
    model.grid_import = pyomo.Var(model.T, within=pyomo.NonNegativeReals)  # kW
    model.grid_export = pyomo.Var(model.T, within=pyomo.NonNegativeReals)  # kW
    # New binary variables (ADD THESE)
    model.diesel_active = pyomo.Var(model.T, within=pyomo.Binary)  # 1 if diesel is used
    model.grid_export_active = pyomo.Var(model.T, within=pyomo.Binary)  # 1 if grid export is used

    # Battery operation
    model.batt_charge = pyomo.Var(model.T, within=pyomo.NonNegativeReals)  # kW
    model.batt_discharge = pyomo.Var(model.T, within=pyomo.NonNegativeReals)  # kW
    model.batt_soc = pyomo.Var(model.T, within=pyomo.NonNegativeReals)  # kWh

    model.battery_full = pyomo.Var(model.T, within=pyomo.Binary)

    # Big-M value (larger than max battery capacity)
    M = 1e6  # Add this near your constants at the top of the file if possible

    # Battery full indicator constraints
    def battery_full_upper(model, t):
     return model.batt_soc[t] <= 0.95*BATTERY_CAPACITY + M*model.battery_full[t]

    def battery_full_lower(model, t):
     return model.batt_soc[t] >= 0.95*BATTERY_CAPACITY - M*(1 - model.battery_full[t])

    model.battery_full_upper = pyomo.Constraint(model.T, rule=battery_full_upper)
    model.battery_full_lower = pyomo.Constraint(model.T, rule=battery_full_lower)


    # Add to Decision Variables section
    model.dump_load = pyomo.Var(model.T, within=pyomo.NonNegativeReals)  # kW

    # Diesel generator (conditional)
    if diesel_enabled:
        model.diesel = pyomo.Var(model.T, within=pyomo.NonNegativeReals)  # kW
    
    # --------------------------
    # Constraints
    # --------------------------
    # Power balance constraint
    def total_cost(model):
        grid_import_cost = sum(model.grid_import[t] * data.loc[t, 'market_price'] * 0.25 for t in model.T)
        grid_export_revenue = sum(model.grid_export[t] * data.loc[t, 'market_price'] * 0.95 * 0.25 for t in model.T)  # 5% grid loss
        batt_cost = sum((model.batt_charge[t] + model.batt_discharge[t]) * BATTERY_COST * 0.25 for t in model.T)
        diesel_cost = sum(model.diesel[t] * DIESEL_COST * 0.25 for t in model.T)
        
        return grid_import_cost - grid_export_revenue + batt_cost + diesel_cost
    
    # No solar curtailment constraint
    def solar_utilization(model, t):
     return model.grid_export[t] + model.batt_charge[t] + model.dump_load[t] == data.loc[t, 'solar']

    # Battery constraints
    def soc_dynamics(model, t):
        if t == 0:
         return model.batt_soc[t] == BATTERY_CAPACITY * 0.5
        else:
         return (model.batt_soc[t] == model.batt_soc[t-1] + 
                (model.batt_charge[t] * 0.95 - model.batt_discharge[t] / 0.95) * 0.25)

def batt_soc_limit(model, t):
    return (0.1 * BATTERY_CAPACITY <= model.batt_soc[t]) & (model.batt_soc[t] <= BATTERY_CAPACITY)


    
    model.soc_dynamics = pyomo.Constraint(model.T, rule=soc_dynamics)
    
    # Battery constraints (split into individual constraints)

    def batt_soc_limit(model, t):
        return model.batt_soc[t] <= BATTERY_CAPACITY

    def batt_charge_limit(model, t):
        return model.batt_charge[t] <= BATTERY_MAX_RATE

    def batt_discharge_limit(model, t):
        return model.batt_discharge[t] <= BATTERY_MAX_RATE

    model.batt_soc_constraint = pyomo.Constraint(model.T, rule=batt_soc_limit)
    model.batt_charge_constraint = pyomo.Constraint(model.T, rule=batt_charge_limit)
    model.batt_discharge_constraint = pyomo.Constraint(model.T, rule=batt_discharge_limit)

    model.battery_low = pyomo.Var(model.T, within=pyomo.Binary)  # 1 when SOC <= 10%

    # Battery low indicator constraints
    def battery_low_indicator(model, t):
     return model.batt_soc[t] <= 0.1*BATTERY_CAPACITY + M*(1 - model.battery_low[t])
    model.battery_low_constr = pyomo.Constraint(model.T, rule=battery_low_indicator)

    
    # Operational Mode Logic Constraints
    M = 1e6  # Large constant for big-M formulation

    # Mode 1: RES meets load, surplus charges battery
    def mode1_preference(model, t):
     return model.batt_charge[t] >= (model.solar_data[t] - model.load_data[t]) * (1 - model.battery_full[t])


# Mode 2: Battery full, excess to dump load
    def mode2_preference(model, t):
        return model.dump_load[t] >= (model.solar_data[t] - model.load_data[t]) * model.battery_full[t]

# Mode 3: RES insufficient, battery discharges
    def mode3_preference(model, t):
        return model.batt_discharge[t] >= (model.load_data[t] - model.solar_data[t]) * (1 - model.battery_low[t])

# Mode 4: Battery depleted, diesel activates
    def mode4_preference(model, t):
        return model.diesel[t] >= (model.load_data[t] - model.solar_data[t]) * model.battery_low[t]

    model.mode1 = pyomo.Constraint(model.T, rule=mode1_preference)
    model.mode2 = pyomo.Constraint(model.T, rule=mode2_preference)
    model.mode3 = pyomo.Constraint(model.T, rule=mode3_preference)
    model.mode4 = pyomo.Constraint(model.T, rule=mode4_preference)

    # Diesel constraints (if enabled)
    if diesel_enabled:
    # Diesel capacity constraint
     def diesel_limit(model, t):
        return model.diesel[t] <= DIESEL_CAPACITY
    model.diesel_limit = pyomo.Constraint(model.T, rule=diesel_limit)

    # New linearized constraints (ADD THESE)
    def diesel_grid_exclusion(model, t):
        return model.diesel_active[t] + model.grid_export_active[t] <= 1
    model.diesel_grid_exclusion = pyomo.Constraint(model.T, rule=diesel_grid_exclusion)

    def diesel_activation(model, t):
        return model.diesel[t] <= DIESEL_CAPACITY * model.diesel_active[t]
    model.diesel_activation = pyomo.Constraint(model.T, rule=diesel_activation)

    def grid_export_activation(model, t):
        return model.grid_export[t] <= GRID_EXPORT_LIMIT * model.grid_export_active[t]
    model.grid_export_activation = pyomo.Constraint(model.T, rule=grid_export_activation)
    
    # --------------------------
    # Objective Function
    # --------------------------
    
    def total_cost(model):
        # Economic objectives
        grid_cost = sum(model.grid_import[t] * data.loc[t, 'market_price'] * 0.25 for t in model.T)
        grid_revenue = sum(model.grid_export[t] * data.loc[t, 'market_price'] * 0.25 for t in model.T)
        batt_cost = sum((model.batt_charge[t] + model.batt_discharge[t]) * BATTERY_COST * 0.25 for t in model.T)
        diesel_cost = sum(model.diesel[t] * DIESEL_COST * 0.25 for t in model.T) if diesel_enabled else 0
    
        # Environmental objectives
        co2_emissions = sum(model.diesel[t] * DIESEL_EMISSION_FACTOR * 0.25 for t in model.T)
        
        # Operational smoothness penalties
        mode_changes = sum(abs(model.operational_mode[t] - model.operational_mode[t-1]) for t in model.T if t > 0)
        
        # Multi-objective weights
        return (0.7*(grid_cost - grid_revenue + batt_cost + diesel_cost) + 
                0.3*co2_emissions + 
                0.05*mode_changes)

    
    return model


# --------------------------
# Post-Processing & Visualization
# --------------------------
def process_results(model, data, diesel_enabled=True):
    """
    Extract results and calculate metrics
    Returns DataFrame with all operational parameters
    """
    results = data.copy()
    results['grid_import'] = [model.grid_import[t].value for t in model.T]
    results['grid_export'] = [model.grid_export[t].value for t in model.T]
    results['batt_charge'] = [model.batt_charge[t].value for t in model.T]
    results['batt_discharge'] = [model.batt_discharge[t].value for t in model.T]
    results['batt_soc'] = [model.batt_soc[t].value for t in model.T]
    
    if diesel_enabled:
        results['diesel'] = [model.diesel[t].value for t in model.T]
    
    # Calculate costs and emissions
    results['grid_cost'] = results['grid_import'] 
    results['batt_cost'] = (results['batt_charge'] + results['batt_discharge']) * BATTERY_COST * 0.25
    results['diesel_cost'] = results['diesel'] * DIESEL_COST * 0.25 if diesel_enabled else 0
    results['diesel_co2'] = results['diesel'] * DIESEL_EMISSION_FACTOR * 0.25 if diesel_enabled else 0
    
    return results

# ADD OPERATIONAL MODE CLASSIFICATION
    results['operational_mode'] = 0
    
    # Mode conditions
    mask_mode1 = (results['solar'] >= results['load']) & (results['batt_charge'] > 0)
    mask_mode2 = (results['batt_soc'] >= 0.95*BATTERY_CAPACITY) & (results['dump_load'] > 0)
    mask_mode3 = (results['solar'] < results['load']) & (results['batt_discharge'] > 0)
    mask_mode4 = (results['batt_soc'] <= 0.1*BATTERY_CAPACITY) & (results['diesel'] > 0)
    
    results.loc[mask_mode1, 'operational_mode'] = 1
    results.loc[mask_mode2, 'operational_mode'] = 2
    results.loc[mask_mode3, 'operational_mode'] = 3
    results.loc[mask_mode4, 'operational_mode'] = 4
    
    return results

def create_visualizations(results, diesel_enabled=True):
    """
    Generate all required visualizations
    """
    import seaborn as sns
    sns.set_theme()

    mode_labels = ['Mode 1 (RES Supply)', 'Mode 2 (Dump Load)', 
                  'Mode 3 (Battery Supply)', 'Mode 4 (Diesel Supply)']
    
    plt.figure(figsize=(15, 6))
    for mode in [1, 2, 3, 4]:
        mode_data = results[results['operational_mode'] == mode]
        plt.scatter(mode_data.index, mode_data['operational_mode'], 
                   label=mode_labels[mode-1], s=50)
    plt.yticks([1,2,3,4], mode_labels)
    plt.title('Operational Modes Over Time')
    plt.legend()
    plt.grid(True)
    # Daily aggregation
    daily = results.resample('D', on='Time').sum()
    
    # --------------------------
    # a) Daily Energy Contribution
    # --------------------------
    plt.figure(figsize=(15, 6))
    components = ['solar', 'batt_discharge']
    if diesel_enabled:
        components.append('diesel')
    components.append('grid_import')

    def process_results(model, data, diesel_enabled=True):
    # ... existing code ...
    

    
        plt.stackplot(daily.index, 
                [daily[c] * 0.25 for c in components],  # Convert kW to kWh
                labels=['Solar', 'Battery', 'Diesel', 'Grid'][:len(components)],
                colors=['gold', 'limegreen', 'brown', 'steelblue'])
    
    plt.title('Daily Energy Contribution')
    plt.ylabel('Energy (kWh)')
    plt.legend(loc='upper left')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.grid(True, alpha=0.3)
    
    # --------------------------
    # b) Daily Cost Comparison
    # --------------------------
    plt.figure(figsize=(15, 6))
    cost_components = ['grid_cost', 'batt_cost']
    if diesel_enabled:
        cost_components.append('diesel_cost')
    
    daily_cost = daily[cost_components]
    daily_cost.plot(kind='bar', stacked=True, 
                   color=['steelblue', 'limegreen', 'brown'][:len(cost_components)])
    
    plt.title('Daily Energy Costs')
    plt.ylabel('Cost ($)')
    plt.xlabel('Date')
    plt.xticks(rotation=45)
    plt.grid(True, axis='y', alpha=0.3)

    
    # d) CO2 Emissions Comparison
    # --------------------------
    if diesel_enabled:
        plt.figure(figsize=(15, 6))
        daily['diesel_co2'].plot(kind='bar', color='brown')
        plt.title('Daily CO2 Emissions from Diesel Generation')
        plt.ylabel('CO2 Emissions (kg)')
        plt.xlabel('Date')
        plt.xticks(rotation=45)
        plt.grid(True, axis='y', alpha=0.3)
    
    plt.show()

# --------------------------
# Main Execution
# --------------------------
if __name__ == "__main__":
    # Load and preprocess data
    data = load_and_preprocess_data(
        'cleaned_solar_data_filled_normalised.csv',
        'chandigarh_power_data_resampled_normalised.csv',
        'cleaned_data5.csv'
    )

    def plot_grid_trading(results):
        plt.figure(figsize=(15, 6))
        plt.plot(results.index, results['grid_import'], label='Grid Import', color='red')
        plt.plot(results.index, -results['grid_export'], label='Grid Export', color='green')
        plt.plot(results.index, results['net_generation'], label='Net Generation', color='blue')
        plt.title('Grid Trading vs Net Generation')
        plt.xlabel('Time')
        plt.ylabel('Power (kW)')
        plt.legend()
        plt.grid(True)
        plt.show()

# Call this function in create_visualizations
        plot_grid_trading(results)

    # Build and solve model
diesel_enabled = True  # Set to False to disable diesel
model = build_optimization_model(data, diesel_enabled)

    
try:
        
        solver = SolverFactory('cbc', executable=cbc_path)
        results = solver.solve(model, tee=True)

        if results.solver.status == pyomo.SolverStatus.ok and results.solver.termination_condition == pyomo.TerminationCondition.optimal:
         print("Optimal solution found")
        else:
         raise RuntimeError("Solver failed. Status: {results.solver.status}, Termination Condition: {results.solver.termination_condition}")
            
except Exception as e:
        print(f"Optimization failed: {str(e)}")
        sys.exit(1)
    
    # Process and visualize results
        results_df = process_results(model, data, diesel_enabled)
        create_visualizations(results_df, diesel_enabled)
    
    # Print summary statistics
        total_cost = results_df[['grid_cost', 'batt_cost', 'diesel_cost']].sum().sum()
        print(f"\nTotal Operational Cost: ${total_cost:.2f}")
    
        if diesel_enabled:
         total_co2 = results_df['diesel_co2'].sum()
         print(f"Total CO2 Emissions: {total_co2:.2f} kg")
