import io
import base64
from datetime import datetime
import pandas as pd
import numpy as np

# Import the simulation function from your uploaded file
from simulation_v2_inflation import run_simulation

# Matplotlib configuration for web serving
import matplotlib
matplotlib.use('Agg') # Use the 'Agg' backend for non-interactive plotting
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates

from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

# Helper function to convert plot to base64 image
def plot_to_base64(fig):
    """Converts a matplotlib figure to a base64 encoded PNG image."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.5)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig) # Close the figure to free up memory
    return img_base64

# Helper function to apply scientific style to plots
def apply_scientific_style(ax, plot_params_text=""):
    """Applies a consistent scientific style to a matplotlib axes object."""
    ax.tick_params(axis='both', which='major', labelsize=10, width=1.5, length=6, direction='inout')
    ax.tick_params(axis='both', which='minor', labelsize=8, width=0.75, length=3, direction='inout')
    ax.minorticks_on()
    ax.grid(True, which='major', linestyle='-', linewidth='0.7', color='lightgray', alpha=0.8)
    ax.grid(True, which='minor', linestyle=':', linewidth='0.5', color='lightgray', alpha=0.5)
    ax.set_title(ax.get_title(), fontsize=14, weight='bold')
    ax.set_xlabel(ax.get_xlabel(), fontsize=12)
    ax.set_ylabel(ax.get_ylabel(), fontsize=12)
    
    # Check if a legend exists before trying to modify it
    if ax.get_legend() is not None:
        ax.legend(fontsize=9, frameon=True, shadow=True, fancybox=True)
    else:
        # If no legend, create a dummy one or handle as needed
        # For stackplot, legend is handled differently, so we might skip this part or add it manually
        pass
        
    # Remove top and right spines for a cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add parameters text to the top-left of the plot if provided
    if plot_params_text:
        ax.text(0.02, 0.98, plot_params_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.6))

@app.route('/')
def index():
    """Renders the main input form page."""
    return render_template('index.html')

@app.route('/run', methods=['POST'])
def run():
    """
    Handles the simulation request, runs the simulation, generates plots,
    and displays results.
    """
    try:
        # Extract form data
        start_date_str = request.form['start_date']
        end_date_str = request.form['end_date']
        initial_cash = float(request.form['initial_cash'])
        invest_frac = float(request.form['invest_frac'])
        threshold_cash = float(request.form['threshold_cash'])
        selected_market_index = request.form['market_index']
        inflation_rate = float(request.form['inflation_rate']) # Make sure to add this to your form

        first_person_job_val = float(request.form['carmen_job_val'])
        first_person_job_start = request.form['carmen_job_start']
        first_person_job_end = request.form['carmen_job_end']
        first_person_rente_val = float(request.form['carmen_rente_val'])
        first_person_rente_start = request.form['carmen_rente_start']

        second_person_job_val = float(request.form['felix_job_val'])
        second_person_job_start = request.form['felix_job_start']
        second_person_job_end = request.form['felix_job_end']
        second_person_rente_val = float(request.form['felix_rente_val'])
        second_person_rente_start = request.form['felix_rente_start']

        large_payment1_val = float(request.form['large_payment1_val'])
        large_payment1_date = request.form['large_payment1_date']
        large_payment2_val = float(request.form['large_payment2_val'])
        large_payment2_date = request.form['large_payment2_date']
        other_monthly_expenses = float(request.form['other_monthly_expenses'])

        # Run the simulation
        t, median_total, df, p5_total, p80_total, prob_below_threshold, prob_zero_cash, \
        cum_income, cum_expense, cum_net, cum_first_person, cum_second_person, \
        median_cash, median_port, cash_paths, port_paths, total_paths, cum_large_payments = \
            run_simulation(start_date_str, end_date_str, initial_cash, invest_frac, threshold_cash,
                           first_person_job_val, first_person_job_start, first_person_job_end,
                           first_person_rente_val, first_person_rente_start,
                           second_person_rente_val, second_person_rente_start,
                           second_person_job_val, second_person_job_start, second_person_job_end,
                           large_payment1_val, large_payment1_date,
                           large_payment2_val, large_payment2_date,
                           other_monthly_expenses, selected_market_index, inflation_rate)

        # Calculate additional parameters for display
        initial_portfolio_value = initial_cash * invest_frac
        median_final_wealth = median_total[-1]
        percentile_5 = np.percentile(total_paths[:, -1], 5)
        inital_monthly_expenses_total = df['expenses'].iloc[0]
        median_final_wealth_after_taxes = median_final_wealth * (1- 0.26)
        
        months = len(df)
        inflation_rate_monthly_factor = (1 + inflation_rate)**(1/12)
        total_inflation_factor = inflation_rate_monthly_factor**months
        median_final_wealth_today_s_power = median_final_wealth_after_taxes / total_inflation_factor
        median_final_wealth_today_s_power_before_taxes = median_final_wealth / total_inflation_factor

        # Text to be added to plots
        plot_params_text = (
            f"Initial Cash: €{initial_cash:,.0f}\n"
            f"Initial Portfolio: €{initial_portfolio_value:,.0f}\n"
            f"Median Final Wealth (inflated): €{median_final_wealth:,.0f}\n"
            f"Median Final Wealth (today's purchasing power): €{median_final_wealth_today_s_power_before_taxes:,.0f}\n"
            f"Median Final Wealth after 26% tax (today's pp): €{median_final_wealth_after_taxes:,.0f} ({median_final_wealth_today_s_power:,.0f})\n"
            f"Market Index: {selected_market_index}\n"
            f"Annual Inflation: {inflation_rate:.1%}\n"
            f"Initial Monthly Expenses: €{inital_monthly_expenses_total:,.0f}\n"
        )

        plots = {}

        # --- Plot 1: Key Financial Trajectories with 5–80%ile Shading ---
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        t_years = t / 12.0
        ax1.plot(t_years, median_cash, label='Median Cash Account', linestyle='--', marker='^', markersize=4, color='green')
        ax1.plot(t_years, median_port, label='Median Portfolio Value', linestyle='-', marker='D', markersize=3, color='purple')
        ax1.fill_between(t_years, p5_total, p80_total, color='orange', alpha=0.3, label='5–80th %ile Total Wealth')
        ax1.plot(t_years, median_total, label='Median Total Wealth', linewidth=2.5, color='orange', linestyle='-')
        ax1.axhline(threshold_cash, color='crimson', linestyle='--', linewidth=1.5, label=f'Cash Threshold €{threshold_cash:,.0f}')
        ax1.set_xlabel('Years since Start Date')
        ax1.set_ylabel('Amount (€)')
        ax1.set_title('Key Financial Trajectories with 5–80%ile Shading')
        apply_scientific_style(ax1, plot_params_text)
        ax1.xaxis.set_major_locator(ticker.MultipleLocator(5))
        ax1.xaxis.set_minor_locator(ticker.MultipleLocator(1))
        plots['plot1'] = plot_to_base64(fig1)

        # --- Plot 2: Cumulative Cash Components Over Time ---
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.plot(t, cum_first_person, label='Cumulative First Person Income', linestyle=':', marker='o', markersize=3, color='darkgreen')
        ax2.plot(t, cum_second_person, label='Cumulative Second Person Income', linestyle='--', marker='s', markersize=3, color='darkblue')
        ax2.plot(t, cum_expense, label='Cumulative Expenses', linestyle='-', color='firebrick')
        ax2.plot(t, cum_large_payments, label='Cumulative Large Payments/Withdrawals', linestyle='-.', marker='x', markersize=4, color='gray')
        ax2.plot(t, cum_income, label = 'Cumulative Total Income', linestyle='-', linewidth=1.5, color='teal')
        ax2.plot(t, cum_net, label= 'Cumulative Net Cash', linewidth = 3.5, color='black', linestyle='-')
        ax2.axhline(threshold_cash, color='red', linestyle='--', linewidth=1.5, label=f'Cash Threshold €{threshold_cash:,.0f}')
        ax2.set_xlabel('Months since Start Date')
        ax2.set_ylabel('Amount (€)')
        ax2.set_title('Cumulative Cash Components Over Time')
        apply_scientific_style(ax2, plot_params_text)
        plots['plot2'] = plot_to_base64(fig2)

        # --- Plot 3: Monthly Cash Flow Components Over Time ---
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        ax3.plot(df.index, df['expenses'], label='Total Monthly Expenses', linestyle='dashed', color='red', marker='v', markersize=2, alpha=0.7)
        ax3.plot(df.index, df['income'], label='Total Monthly Income', linestyle='dashed', color='green', marker='^', markersize=2, alpha=0.7)
        ax3.plot(df.index, df['net_cash'], label='Monthly Net Cash', linestyle='solid', color='blue', linewidth=1.5)
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Amount (€ / month)')
        ax3.set_title('Monthly Cash Flow Components Over Time')
        apply_scientific_style(ax3, plot_params_text)
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        fig3.autofmt_xdate()
        plots['plot3'] = plot_to_base64(fig3)

        # --- Plot 4: Histogram of Final Total Wealth ---
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        final = total_paths[:, -1]
        ax4.hist(final, bins=40, density=True, alpha=0.8, color='skyblue', edgecolor='black')
        ax4.axvline(median_final_wealth, color='red', linestyle='--', label=f'Median Final Wealth: €{median_final_wealth:,.0f}')
        ax4.axvline(percentile_5, color = 'black', label= f'5th Percentile: €{percentile_5:,.0f}', linestyle='--')
        ax4.set_xlabel('Final Total Wealth (€)')
        ax4.set_ylabel('Probability Density')
        ax4.set_title(f'Histogram of Final Total Wealth ({df.index[-1].strftime("%b %Y")})')
        apply_scientific_style(ax4, plot_params_text)
        plots['plot4'] = plot_to_base64(fig4)

        # --- Plot 5: Probability of Cash Account Reaching Critical Levels Over Time ---
        fig5, ax5 = plt.subplots(figsize=(10, 6))
        ax5.plot(t, prob_below_threshold, label=f'Probability Cash < €{threshold_cash:,.0f}', color='blue', linewidth=2, marker='.')
        ax5.plot(t, prob_zero_cash, label='Probability Cash <= €0', color='red', linestyle='--', linewidth=2, marker='x')
        ax5.set_xlabel('Months since Start Date')
        ax5.set_ylabel('Probability')
        ax5.set_title('Probability of Cash Account Reaching Critical Levels Over Time')
        ax5.set_ylim(0, 1)
        apply_scientific_style(ax5, plot_params_text)
        plots['plot5'] = plot_to_base64(fig5) # This was plot5 in the original, but results.html refers to it as plot3, so I will map it accordingly

        # --- Plot 6: Median Total Wealth Breakdown: Cash vs. Portfolio (Stacked Area Plot) ---
        fig6, ax6 = plt.subplots(figsize=(10, 6))
        ax6.stackplot(t, median_cash, median_port, labels=['Median Cash Account', 'Median Portfolio Value'], alpha=0.8, colors=['lightgreen', 'lightblue'])
        ax6.plot(t, median_total, color='black', linestyle='--', linewidth=2, label='Median Total Wealth')
        ax6.set_xlabel('Months since Start Date')
        ax6.set_ylabel('Amount (€)')
        ax6.set_title('Median Total Wealth Breakdown: Cash vs. Portfolio')
        # Manually add legend for stackplot as apply_scientific_style might override
        ax6.legend(loc='upper left', fontsize=9, frameon=True, shadow=True, fancybox=True) 
        # Hide the generic text added by apply_scientific_style and add it back manually if needed
        # apply_scientific_style(ax6, plot_params_text) # This will add the text twice if not careful
        #ax6.text(0.02, 0.98, plot_params_text, transform=ax6.transAxes,
         #           fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.6))
        plots['plot6'] = plot_to_base64(fig6)

        # --- Plot 7: Annual Net Cash Flow (Bar Chart) ---
        fig7, ax7 = plt.subplots(figsize=(12, 6))
        df_annual = df.resample('Y').sum(numeric_only=True)
        annual_net_cash = df_annual['net_cash']
        annual_labels = [str(x.year) for x in df_annual.index]
        bars = ax7.bar(annual_labels, annual_net_cash, color=['skyblue' if x >= 0 else 'lightcoral' for x in annual_net_cash], edgecolor='black', linewidth=0.7)
        ax7.axhline(0, color='red', linestyle='--', linewidth=1.5)
        ax7.set_xlabel('Year')
        ax7.set_ylabel('Annual Net Cash Flow (€)')
        ax7.set_title('Annual Net Cash Flow (Income - Expenses + Large Payments)')
        ax7.tick_params(axis='x', rotation=45) # Removed ha='right'
        apply_scientific_style(ax7, plot_params_text)
        for bar in bars:
            yval = bar.get_height()
            ax7.text(bar.get_x() + bar.get_width()/2, yval + (5000 if yval >= 0 else -15000), f'€{yval:,.0f}',
                     ha='center', va='bottom' if yval >= 0 else 'top', fontsize=8, color='black')
        plots['plot7'] = plot_to_base64(fig7)
        
        # --- Plot 8: Cumulative Impact of Large Financial Events ---
        fig8, ax8 = plt.subplots(figsize=(10, 6))
        ax8.plot(t, cum_large_payments, label='Cumulative Large Payments/Withdrawals', color='purple', linewidth=2, marker='.', markersize=4)
        ax8.set_xlabel('Months since Start Date')
        ax8.set_ylabel('Amount (€)')
        ax8.set_title('Cumulative Impact of Large Financial Events')
        apply_scientific_style(ax8, plot_params_text)
        plots['plot8'] = plot_to_base64(fig8)

        return render_template('results.html', plots=plots, final_wealth=f"€{median_final_wealth_today_s_power_before_taxes:,.0f}", final_wealth_after_tax = f"€{median_final_wealth_after_taxes}")

    except ValueError as e:
        # Handle invalid input errors
        return render_template('index.html', error=f"Input Error: {e}")
    except Exception as e:
        # Handle any other unexpected errors
        return render_template('index.html', error=f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    # This block is for local development. In a Canvas environment, the app is run differently.
    # To run locally, create a 'templates' folder and put index.html and results.html inside.
    # Create a 'static' folder and put placeholder images (or your actual images) inside.
    # You might need to install Flask: pip install Flask
    # Then run: python app.py
    app.run(debug=True) # For local debugging
