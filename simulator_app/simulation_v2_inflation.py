import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def run_simulation(start_date_str, end_date_str, initial_cash, invest_frac, threshold_cash,
                   first_person_job_val, first_person_job_start, first_person_job_end,
                   first_person_rente_val, first_person_rente_start,
                   second_person_rente_val, second_person_rente_start,
                   second_person_job_val, second_person_job_start, second_person_job_end,
                   large_payment1_val, large_payment1_date,
                   large_payment2_val, large_payment2_date,
                   other_monthly_expenses, selected_market_index, inflation_rate):
    try:
        # PARAMETERS
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d')

        # Annual discrete return distribution
        sp_annual_returns = np.array([-0.35, -0.20, -0.10, -0.05, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
        sp_probs = np.array([0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.18, 0.12, 0.07, 0.03])
        
        eurostoxx_annual_returns = np.array([-0.40, -0.25, -0.12, -0.06, 0.04, 0.09, 0.14, 0.18, 0.22, 0.28])
        eurostoxx_probs = np.array([0.03, 0.06, 0.10, 0.12, 0.14, 0.18, 0.15, 0.10, 0.08, 0.04])
        
        #monthly inflation (inflation_rate factor must be yearly):
        inflation_rate_monthly = (1 + inflation_rate)**(1/12)

        if selected_market_index == "S&P 500":
            annual_returns = sp_annual_returns
            probs = sp_probs
        elif selected_market_index == "Euro Stoxx 50":
            annual_returns = eurostoxx_annual_returns
            probs = eurostoxx_probs
        else:
            raise ValueError("Invalid market index selected.")
            
        monthly_returns = (1 + annual_returns)**(1/12) - 1

        # BUILD MONTHLY CASH-FLOW DF
        df = pd.DataFrame(index=pd.date_range(start_date, end_date, freq='M'))
        
        # Initialize original fixed expenses
        original_fixed_expenses = {}
        for col,val in [('miete',2500),('essen',500),('auto',100),
                        ('kvg',1300),('strom',100),('freizeit',200)]:
            original_fixed_expenses[col] = val
            df[col] = val
        
        # Initialize original job and rente values
        original_first_person_job_val = first_person_job_val
        original_first_person_rente_val = first_person_rente_val
        original_second_person_job_val = second_person_job_val
        original_second_person_rente_val = second_person_rente_val

        # Apply inflation to expenses and income over time
        for i in range(len(df)):
            inflation_factor = inflation_rate_monthly**i
            for col, val in original_fixed_expenses.items():
                df.loc[df.index[i], col] = val * inflation_factor
            
            # Apply inflation to other monthly expenses
            df.loc[df.index[i], 'other_monthly_expenses_inflated'] = other_monthly_expenses * inflation_factor
        
        df['fixed_expenses'] = df[['miete','essen','auto','kvg','strom','freizeit']].sum(axis=1)
        df['expenses'] = df['fixed_expenses'] + df['other_monthly_expenses_inflated']

        # Add randomness to monthly expenses (10% of mean fluctuation)
        random_factors = np.random.uniform(0.95, 1.05, size=len(df))
        df['expenses'] = df['expenses']# * random_factors

        def H(val, s, e, inflation_series):
            # Convert start and end dates to datetime objects if they are strings
            s = datetime.strptime(s, '%Y-%m-%d') if isinstance(s, str) else s
            e = datetime.strptime(e, '%Y-%m-%d') if isinstance(e, str) else e
            
            # Create a series with the base value
            base_val_series = pd.Series(0.0, index=df.index)
            # Apply base value only within the specified date range
            base_val_series.loc[(base_val_series.index >= s) & (base_val_series.index <= e + pd.DateOffset(months=1, days=-1))] = val

            # Multiply by the inflation series
            return base_val_series * inflation_series


        # Generate inflation factors for income
        income_inflation_factors = pd.Series([inflation_rate_monthly**i for i in range(len(df))], index=df.index)

        # Apply user-defined income streams with inflation
        df['first_person_job']   = H(original_first_person_job_val, first_person_job_start, first_person_job_end, income_inflation_factors)
        df['first_person_rente'] = H(original_first_person_rente_val, first_person_rente_start, end_date_str, income_inflation_factors)
        df['second_person_job']    = H(original_second_person_job_val, second_person_job_start, second_person_job_end, income_inflation_factors)
        df['second_person_rente']  = H(original_second_person_rente_val, second_person_rente_start, end_date_str, income_inflation_factors)
        df['income']       = df[['first_person_job','first_person_rente','second_person_job','second_person_rente']].sum(axis=1)

        # Add large payments/withdrawals (these are assumed to be in today's terms and not inflated)
        df['large_payments'] = 0.0
        if large_payment1_val != 0:
            df.loc[pd.to_datetime(large_payment1_date).to_period('M').to_timestamp('M'), 'large_payments'] += large_payment1_val
        if large_payment2_val != 0:
            df.loc[pd.to_datetime(large_payment2_date).to_period('M').to_timestamp('M'), 'large_payments'] += large_payment2_val

        df['net_cash']     = df['income'] - df['expenses'] + df['large_payments']
        
        # cumulative monthly incomes & expenses (calculated as before, but now returned)
        cum_income  = np.concatenate(([0], df['income'].cumsum().values))
        cum_expense = np.concatenate(([0], df['expenses'].cumsum().values))
        cum_first_person = np.concatenate(([0], (df['first_person_job']+ df['first_person_rente']).cumsum().values))
        cum_second_person = np.concatenate(([0], (df['second_person_job']+ df['second_person_rente']).cumsum().values))
        cum_large_payments = np.concatenate(([0], df['large_payments'].cumsum().values)) # Although not plotted here, useful to return
        cum_net = np.concatenate(([0], df['net_cash'].cumsum().values))
        
        # MONTE CARLO SIMULATION at MONTHLY STEPS
        months          = len(df)
        initial_port    = initial_cash * invest_frac
        initial_cash_ac = initial_cash * (1 - invest_frac)

        n_sims = 200
        cash_paths  = np.zeros((n_sims, months+1))
        port_paths  = np.zeros((n_sims, months+1))
        total_paths = np.zeros((n_sims, months+1))
        cash_below_threshold_count = np.zeros(months + 1)
        cash_zero_count = np.zeros(months + 1)

        for sim in range(n_sims):
            cash  = np.zeros(months+1)
            port  = np.zeros(months+1)
            cash[0], port[0] = initial_cash_ac, initial_port

            for t_idx in range(1, months+1):
                r = np.random.choice(monthly_returns, p=probs)
                port[t_idx] = port[t_idx-1] * (1 + r)
                
                current_net_cash_flow_without_large_payment = df['income'].iloc[t_idx-1] - df['expenses'].iloc[t_idx-1]
                large_payment_val = df['large_payments'].iloc[t_idx-1]

                if large_payment_val < 0:
                    cash[t_idx] = cash[t_idx-1] + current_net_cash_flow_without_large_payment + large_payment_val
                    if cash[t_idx] < threshold_cash:
                        shortfall = threshold_cash - cash[t_idx]
                        port[t_idx] -= shortfall
                        cash[t_idx] = threshold_cash
                elif large_payment_val > 0:
                    cash[t_idx] = cash[t_idx-1] + current_net_cash_flow_without_large_payment
                    port[t_idx] += large_payment_val
                else:
                    cash[t_idx] = cash[t_idx-1] + current_net_cash_flow_without_large_payment
                if large_payment_val >= 0 and cash[t_idx] < threshold_cash:
                    shortfall = threshold_cash - cash[t_idx]
                    port[t_idx]  -= shortfall
                    cash[t_idx]  = threshold_cash

                if cash[t_idx] < threshold_cash:
                    cash_below_threshold_count[t_idx] += 1
                if cash[t_idx] <= 0:
                    cash_zero_count[t_idx] += 1

            cash_paths[sim]  = cash
            port_paths[sim]  = port
            total_paths[sim] = cash + port

        # compute median and percentiles
        t = np.arange(months+1)
        median_cash  = np.median(cash_paths, axis=0)
        median_port  = np.median(port_paths, axis=0)
        median_total = np.median(total_paths, axis=0)
        p5_total    = np.percentile(total_paths, 5, axis=0)
        p80_total    = np.percentile(total_paths, 80, axis=0)

        prob_below_threshold = cash_below_threshold_count / n_sims
        prob_zero_cash = cash_zero_count / n_sims
        
        # Return all necessary data for plotting and display
        return t, median_total, df, p5_total, p80_total, prob_below_threshold, prob_zero_cash, \
               cum_income, cum_expense, cum_net, cum_first_person, cum_second_person, median_cash, median_port, cash_paths, port_paths, total_paths, cum_large_payments

    except Exception as e:
        raise ValueError(f"Simulation Error: {e}")