# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pyomo.environ as pyomo
from pyomo.opt import SolverFactory
import sys
import numpy as np

# --------------------------
# Configuration Parameters
# --------------------------
SOLAR_CAPACITY = 5000  # kW (2.5 MW)
BATTERY_CAPACITY = 15000  # kWh (5 MWh)
BATTERY_MAX_RATE = 5000  # kW (2.5 MW charge/discharge rate from specs)
GRID_EXPORT_LIMIT = 3000  # kW (adjust based on your grid connection capacity)
GRID_IMPORT_LIMIT = 3000
# --------------------------
# Data Loading & Preprocessing
# --------------------------
def load_and_preprocess_data(solar_file, load_file):
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
            parse_dates=['Timestamp'],
            index_col='Timestamp',
            usecols=['Timestamp', 'VALUE'],
            dtype={'VALUE': float}
        ).rename(columns={'VALUE': 'load'})

        # Convert MW to kW
        solar_df['solar'] *= 1000
        load_df['load'] *= 1000

        # Merge data using inner join to ensure alignment
        df = pd.merge(
            solar_df,
            load_df,
            left_index=True,
            right_index=True,
            how='inner'
        )

        # Validate data
        if df.isnull().sum().sum() > 0:
            df = df.ffill().bfill()
            print("Warning: Missing values filled using forward/backward fill")
        print(f"Loaded data with {len(df)} time points")

        return df.reset_index().rename(columns={'index': 'Timestamp'})

    except Exception as e:
        print(f"Data loading failed: {str(e)}")
        sys.exit(1)

# --------------------------
# Pyomo Optimization Model
# --------------------------
def build_optimization_model(data):
    """
    Build Pyomo optimization model focused on load balancing
    """
    model = pyomo.ConcreteModel()
    
    # Time periods (15-min intervals)
    model.T = pyomo.Set(initialize=data.index)
    
    # --------------------------
    # Decision Variables
    # --------------------------
    # Grid interaction
    model.grid_import = pyomo.Var(model.T, within=pyomo.NonNegativeReals)  # kW
    model.grid_export = pyomo.Var(model.T, within=pyomo.NonNegativeReals)  # kW
    
    # Battery operation
    model.batt_charge = pyomo.Var(model.T, within=pyomo.NonNegativeReals)  # kW
    model.batt_discharge = pyomo.Var(model.T, within=pyomo.NonNegativeReals)  # kW
    model.batt_soc = pyomo.Var(model.T, within=pyomo.NonNegativeReals)  # kWh
    
    # Load balancing variables
    model.net_load = pyomo.Var(model.T)  # Can be positive or negative
    
    # --------------------------
    # Constraints
    # --------------------------
    # Power balance constraint
    def power_balance(model, t):
        solar = data.loc[t, 'solar']
        load = data.loc[t, 'load']
        
        return (solar + model.grid_import[t] + model.batt_discharge[t] == 
                load + model.grid_export[t] + model.batt_charge[t])
    
    model.power_balance = pyomo.Constraint(model.T, rule=power_balance)
    
    # Net load calculation (load - solar)
    def net_load_calculation(model, t):
        return model.net_load[t] == data.loc[t, 'load'] - data.loc[t, 'solar']
    
    model.net_load_calc = pyomo.Constraint(model.T, rule=net_load_calculation)
    
    # Battery constraints
    def soc_dynamics(model, t):
        if t == 0:
            return model.batt_soc[t] == BATTERY_CAPACITY * 0.5  # Start at 50% SOC
        else:
            return model.batt_soc[t] == (model.batt_soc[t-1] + 
                                       model.batt_charge[t] * 0.25 -  # 15-min to hours
                                       model.batt_discharge[t] * 0.25)
    
    model.soc_dynamics = pyomo.Constraint(model.T, rule=soc_dynamics)
    
    # Final SOC constraint (end at 50% capacity)
    def final_soc(model):
        t_final = model.T.last()
        return model.batt_soc[t_final] == BATTERY_CAPACITY * 0.5
    
    model.final_soc = pyomo.Constraint(rule=final_soc)
    
    # Battery capacity constraints
    def batt_soc_limit(model, t):
        return model.batt_soc[t] <= BATTERY_CAPACITY

    def batt_charge_limit(model, t):
        return model.batt_charge[t] <= BATTERY_MAX_RATE

    def batt_discharge_limit(model, t):
        return model.batt_discharge[t] <= BATTERY_MAX_RATE

    model.batt_soc_constraint = pyomo.Constraint(model.T, rule=batt_soc_limit)
    model.batt_charge_constraint = pyomo.Constraint(model.T, rule=batt_charge_limit)
    model.batt_discharge_constraint = pyomo.Constraint(model.T, rule=batt_discharge_limit)
    
    # Prevent simultaneous charging and discharging
    def no_simul_charge_discharge(model, t):
        return model.batt_charge[t] * model.batt_discharge[t] == 0
    
    # Note: This is a non-linear constraint that won't work with GLPK
    # We'll use binary variables instead for a linear formulation
    
    model.batt_charge_binary = pyomo.Var(model.T, within=pyomo.Binary)
    
    # Battery can either charge or discharge but not both
    def charge_discharge_exclusivity1(model, t):
        return model.batt_charge[t] <= BATTERY_MAX_RATE * model.batt_charge_binary[t]
    
    def charge_discharge_exclusivity2(model, t):
        return model.batt_discharge[t] <= BATTERY_MAX_RATE * (1 - model.batt_charge_binary[t])
    
    model.charge_excl_1 = pyomo.Constraint(model.T, rule=charge_discharge_exclusivity1)
    model.charge_excl_2 = pyomo.Constraint(model.T, rule=charge_discharge_exclusivity2)
    
    # Grid export limit
    def grid_export_limit(model, t):
        return model.grid_export[t] <= GRID_EXPORT_LIMIT
    
    model.grid_export_constr = pyomo.Constraint(model.T, rule=grid_export_limit)
    
    # --------------------------
    # Objective Function: Minimize grid usage and stabilize load
    # --------------------------
    def system_stability(model):
        # Calculate the net power after battery contribution
        grid_dependency = sum(model.grid_import[t] + model.grid_export[t] for t in model.T)
        
        # Minimize grid dependency
        return grid_dependency
    
    model.obj = pyomo.Objective(rule=system_stability, sense=pyomo.minimize)
    
    return model

# --------------------------
# Post-Processing & Visualization
# --------------------------
def process_results(model, data):
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
    
    # Calculate net load (load - solar)
    results['net_load'] = results['load'] - results['solar']
    
    # Calculate total supply (solar + battery discharge + grid import)
    results['total_supply'] = (
        results['solar'] + 
        results['batt_discharge'] - 
        results['batt_charge'] + 
        results['grid_import'] - 
        results['grid_export']
    )
    
    # Calculate battery net contribution (discharge - charge)
    results['battery_net'] = results['batt_discharge'] - results['batt_charge']
    
    # Grid net (import - export)
    results['grid_net'] = results['grid_import'] - results['grid_export']
    
    return results

def create_visualizations(results):
    """
    Generate all required visualizations
    """
    import seaborn as sns
    sns.set_theme(style="whitegrid")
    print("Available columns:", results.columns.tolist())
    
    # Try to use the first column as the time index
    # Most likely, it's called 'Timestamp' or something similar
    time_column = results.columns[0]  # Assuming first column is timestamp
    times = pd.to_datetime(results[time_column])
    # Set up plotting parameters
    plt.rcParams.update({'font.size': 12})
    colors = {
        'solar': '#FFD700',       # Gold
        'load': '#1E90FF',        # Dodger Blue
        'battery_pos': '#32CD32', # Lime Green
        'battery_neg': '#FF6347', # Tomato
        'grid': '#4B0082'         # Indigo
    }
    
    # Calculate time points for x-axis formatting
    times = pd.to_datetime(results['Time'])
    x_ticks = np.arange(0, len(results), len(results)//10)  # ~10 tick marks
    
    # --------------------------
    # 1. Solar vs Load Mismatch (15-minute intervals)
    # --------------------------
    plt.figure(figsize=(15, 6))
    plt.plot(times, results['solar'], label='Solar Generation', color=colors['solar'], linewidth=2)
    plt.plot(times, results['load'], label='Load Demand', color=colors['load'], linewidth=2)
    plt.fill_between(times, results['solar'], results['load'], 
                    where=(results['solar'] < results['load']),
                    color='red', alpha=0.3, label='Generation Deficit')
    plt.fill_between(times, results['solar'], results['load'], 
                    where=(results['solar'] > results['load']),
                    color='green', alpha=0.3, label='Generation Surplus')
    
    plt.title('Solar Generation vs Load Demand Mismatch (15-minute intervals)')
    plt.ylabel('Power (kW)')
    plt.xlabel('Time')
    plt.xticks(times[x_ticks], times[x_ticks].dt.strftime('%m-%d %H:%M'), rotation=45)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.grid(True, alpha=0.3)
    
    # --------------------------
    # 2. Battery Charging/Discharging and State of Charge (15-minute intervals)
    # --------------------------
    fig, ax1 = plt.subplots(figsize=(15, 6))
    
    # Battery charge/discharge
    batt_charge = -results['batt_charge']  # Negate to show as negative values
    batt_discharge = results['batt_discharge']
    batt_net = np.where(results['battery_net'] > 0, results['battery_net'], 0)
    batt_net_neg = np.where(results['battery_net'] < 0, results['battery_net'], 0)
    
    ax1.bar(times, batt_net, color=colors['battery_pos'], label='Battery Discharge', alpha=0.7)
    ax1.bar(times, batt_net_neg, color=colors['battery_neg'], label='Battery Charge', alpha=0.7)
    ax1.set_ylabel('Power (kW)')
    ax1.set_title('Battery Operation (15-minute intervals)')
    
    # State of charge on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(times, results['batt_soc'], color='black', linewidth=2, label='State of Charge')
    ax2.set_ylabel('Battery State of Charge (kWh)')
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
    
    plt.xlabel('Time')
    plt.xticks(times[x_ticks], times[x_ticks].dt.strftime('%m-%d %H:%M'), rotation=45)
    plt.tight_layout()
    plt.grid(True, alpha=0.3)
    
    # --------------------------
    # 3. System Stabilization: Before and After Battery (15-minute intervals)
    # --------------------------
    plt.figure(figsize=(15, 6))
    
    # Before battery: Just net load (load - solar)
    plt.plot(times, results['net_load'], label='Net Load (Load - Solar)', color='red', 
            linewidth=2, linestyle='--')
    
    # After battery: Net load with battery contribution
    stabilized_load = results['net_load'] - results['battery_net']
    plt.plot(times, stabilized_load, label='Stabilized Load (after battery)', 
            color='blue', linewidth=2)
    
    plt.title('Load Stabilization with Battery (15-minute intervals)')
    plt.ylabel('Net Power (kW)')
    plt.xlabel('Time')
    plt.xticks(times[x_ticks], times[x_ticks].dt.strftime('%m-%d %H:%M'), rotation=45)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.grid(True, alpha=0.3)
    
    # --------------------------
    # 4. Full Energy Flow Breakdown (15-minute intervals)
    # --------------------------
    plt.figure(figsize=(15, 7))
    
    # Stacked plot showing all energy components
    plt.stackplot(times, 
                 results['solar'],
                 results['battery_net'],
                 results['grid_net'],
                 labels=['Solar', 'Battery Net', 'Grid Net'],
                 colors=[colors['solar'], colors['battery_pos'], colors['grid']])
    
    plt.plot(times, results['load'], label='Load', color='black', linewidth=2)
    
    plt.title('Complete Energy Flow Breakdown (15-minute intervals)')
    plt.ylabel('Power (kW)')
    plt.xlabel('Time')
    plt.xticks(times[x_ticks], times[x_ticks].dt.strftime('%m-%d %H:%M'), rotation=45)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.grid(True, alpha=0.3)
    
    plt.show()

# --------------------------
# Main Execution
# --------------------------
if __name__ == "__main__":
    # Load and preprocess data
    data = load_and_preprocess_data(
        'cleaned_solar_data_filled_normalised.csv',
        'modified_chandigarh_power_data.csv'
    )
    
    # Build and solve model
    model = build_optimization_model(data)
    
    try:
        solver = SolverFactory('glpk')
        results = solver.solve(model, tee=True)

        if results.solver.status == pyomo.SolverStatus.ok and results.solver.termination_condition == pyomo.TerminationCondition.optimal:
            print("Optimal solution found")
        else:
            raise RuntimeError(f"Solver failed. Status: {results.solver.status}, Termination Condition: {results.solver.termination_condition}")
            
    except Exception as e:
        print(f"Optimization failed: {str(e)}")
        sys.exit(1)
    
    # Process and visualize results
    results_df = process_results(model, data)
    create_visualizations(results_df)
    
    # Print summary statistics
    total_grid_import = results_df['grid_import'].sum() * 0.25  # kWh
    total_grid_export = results_df['grid_export'].sum() * 0.25  # kWh
    total_solar = results_df['solar'].sum() * 0.25  # kWh
    total_load = results_df['load'].sum() * 0.25  # kWh
    battery_throughput = (results_df['batt_charge'].sum() + results_df['batt_discharge'].sum()) * 0.25  # kWh
    
    print("\nSUMMARY STATISTICS:")
    print(f"Total Load: {total_load:.2f} kWh")
    print(f"Total Solar Generation: {total_solar:.2f} kWh")
    print(f"Total Grid Import: {total_grid_import:.2f} kWh")
    print(f"Total Grid Export: {total_grid_export:.2f} kWh")
    print(f"Battery Throughput: {battery_throughput:.2f} kWh")
    print(f"Solar Self-Consumption: {(total_solar - total_grid_export) / total_solar * 100:.2f}%")
    print(f"Load Self-Sufficiency: {(total_load - total_grid_import) / total_load * 100:.2f}%")