import io
import base64
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.gridspec as gridspec # Add this import at the top of main_app.py
from html.parser import HTMLParser
from simulator_app.simulation_v2_inflation import run_simulation

import matplotlib
matplotlib.use('Agg') # Use the 'Agg' backend for non-interactive plotting
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates

import pickle # For serializing Python objects
import os     # For path manipulation and file cleanup
import uuid   # To generate unique filenames

from flask import Flask, render_template, request, redirect, url_for, session, send_file


app = Flask(__name__)
app.secret_key = 'some_key'

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



def generate_summary_text(median_final_wealth, median_final_wealth_after_taxes, 
                          median_final_wealth_today_s_power_before_taxes, median_final_wealth_today_s_power,
                          prob_zero_cash_final,
                          start_date_str, end_date_str, selected_market_index, inflation_rate, percentile_90_wealth):
    """Generates a concise summary of the simulation results."""
    
    # Get the start and end years for better context
    start_year = datetime.strptime(start_date_str, '%Y-%m-%d').year
    end_year = datetime.strptime(end_date_str, '%Y-%m-%d').year

    summary = (
        f"🌟 <strong>Simulation Summary ({start_year}-{end_year})</strong> 🌟<br><br>"
        f"This financial simulation, based on the <strong>{selected_market_index}</strong> market index and an "
        f"annual inflation rate of <strong>{inflation_rate:.1%}</strong>, projects your wealth trajectory.<br><br>"
        f"📊 <strong>Key Outcomes:</strong><br>"
        f"   - Median Final Wealth (Inflated): €<strong>{median_final_wealth:,.0f}</strong><br>"
        f"   which in today's Purchasing Power accounts for : €<strong>{median_final_wealth_today_s_power_before_taxes:,.0f}</strong><br>"
        f"   - Median Final Wealth after Tax (Today's Purchasing Power): €<strong>{median_final_wealth_after_taxes:,.0f} ({median_final_wealth_today_s_power:,.0f}</strong>)<br><br>"
        f"📈 <strong>Risk Assessment:</strong><br>"
        #f"   - The probability of your cash account reaching zero by the end of the simulation is <strong>{prob_zero_cash_final:.1%}</strong>.<br><br>" # Updated this line
        f"   - With <strong>90% certainty</strong>, you will reach a wealth of at least <strong>€{percentile_90_wealth:,.0f}</strong>.<br><br>" # New line
        f"This simulation provides a robust outlook on your financial future under various market conditions, adjusted for inflation and potential tax implications."
    )
    return summary

def generate_pdf_report(figures_dict, summary_html, output_path):
    with PdfPages(output_path) as pdf:
        # Add a title page with the summary
        fig_summary = plt.figure(figsize=(8.5, 11)) # A4 size
        ax_summary = fig_summary.add_subplot(111)
        ax_summary.text(0.5, 0.95, "Financial Simulation Report",
                        fontsize=20, ha='center', va='top', transform=ax_summary.transAxes)

        # Helper to strip HTML tags for plain text in PDF
        class MLStripper(HTMLParser):
            def __init__(self):
                super().__init__()
                self.reset()
                self.strict = False
                self.convert_charrefs = True
                self.text = []

            def handle_data(self, d):
                self.text.append(d)

            def get_data(self):
                return ''.join(self.text)

        def strip_tags_and_format(html_text):
            s = MLStripper()
            s.feed(html_text)
            plain_text = s.get_data()
            plain_text = plain_text.replace("</p>", "\n\n").replace("</li>", "\n").replace("<ul>", "").replace("<li>", "- ")
            return plain_text

        plain_text_summary = strip_tags_and_format(summary_html)

        ax_summary.text(0.05, 0.9, plain_text_summary,
                        fontsize=10, ha='left', va='top', transform=ax_summary.transAxes,
                        wrap=True) # wrap=True is important for long text
        ax_summary.axis('off') # Hide axes for text page
        pdf.savefig(fig_summary, bbox_inches='tight')
        plt.close(fig_summary) # Close the figure to free memory

        # Add each plot to the PDF
        for plot_name, fig in figures_dict.items():
            pdf.savefig(fig, bbox_inches='tight') # Save each figure to a new page
            plt.close(fig) # Close the figure after saving to PDF



# Define route for the main website's index page
@app.route('/')
def main_index():
    return render_template('main_index.html', active_page='home')

@app.route('/simulator')
def simulator_index():
    return render_template('simulator_parameters.html', active_page='simulator')

@app.route('/contact')
def contact():
    return render_template('contact.html', active_page='contact')

@app.route('/simulator/documentation')
def simulator_documentation():
    # This would be a new page for your documentation
    return render_template('documentation.html', active_page='documentation') # You'll need to create documentation.html



@app.route('/simulator/run', methods=['POST'])
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
        initial_monthly_expenses = float(request.form['initial_monthly_expenses']) 

        large_payment1_val = float(request.form['large_payment1_val'])
        large_payment1_date = request.form['large_payment1_date']
        large_payment2_val = float(request.form['large_payment2_val'])
        large_payment2_date = request.form['large_payment2_date']
        #other_monthly_expenses = float(request.form['other_monthly_expenses'])

        # Run the simulation
        t, median_total, df, p5_total, p80_total, prob_below_threshold, prob_zero_cash, \
        cum_income, cum_expense, cum_net, cum_first_person, cum_second_person, \
        median_cash, median_port, cash_paths, port_paths, total_paths, cum_large_payments = \
            run_simulation(start_date_str, end_date_str, initial_cash, invest_frac, threshold_cash,
                           first_person_job_val, first_person_job_start, first_person_job_end,
                           first_person_rente_val, first_person_rente_start,
                           second_person_rente_val, second_person_rente_start,
                           second_person_job_val, second_person_job_start, second_person_job_end, initial_monthly_expenses,
                           large_payment1_val, large_payment1_date,
                           large_payment2_val, large_payment2_date,
                           selected_market_index, inflation_rate)

        # Calculate additional parameters for display
        initial_portfolio_value = initial_cash * invest_frac
        median_final_wealth = median_total[-1]
        percentile_5 = np.percentile(total_paths[:, -1], 5)
        inital_monthly_expenses_total = initial_monthly_expenses
        median_final_portfolio_value = median_port[-1]
        portfolio_gain = median_final_portfolio_value - initial_portfolio_value   #streng genommen müsste hier der Gain seit letzter Verkauf oder? 

        
        if portfolio_gain > 0: # Apply tax only if there's a positive gain
            tax_on_gain = portfolio_gain * 0.26
            # Subtract tax from the overall median final wealth
            median_final_wealth_after_taxes = median_final_wealth - tax_on_gain
        else:
            median_final_wealth_after_taxes = median_final_wealth # No tax if no gain or a loss in portfolio
        # --- MODIFICATION END ---
        
        months = len(df)
        inflation_rate_monthly_factor = (1 + inflation_rate)**(1/12)
        total_inflation_factor = inflation_rate_monthly_factor**months
        median_final_wealth_today_s_power = median_final_wealth_after_taxes / total_inflation_factor
        median_final_wealth_today_s_power_before_taxes = median_final_wealth / total_inflation_factor
        prob_zero_cash_final = prob_zero_cash[-1]
        percentile_90_wealth = np.percentile(total_paths[:, -1], 10)
        simulation_summary = generate_summary_text(
            median_final_wealth, median_final_wealth_after_taxes, 
            median_final_wealth_today_s_power_before_taxes, median_final_wealth_today_s_power,
            prob_zero_cash_final,
            start_date_str, end_date_str, selected_market_index, inflation_rate, percentile_90_wealth
        )




        # Text to be added to plots
        plot_params_text = (
            f"Initial Cash: €{initial_cash:,.0f}\n"
            f"Initial Portfolio: €{initial_portfolio_value:,.0f}\n"
            f"Median Final Wealth (today's purchasing power): €{median_final_wealth:,.0f} ({median_final_wealth_today_s_power_before_taxes:,.0f})\n"
            f"Median Final Wealth after cap-gains-tax (today's purchasing power): €{median_final_wealth_after_taxes:,.0f} ({median_final_wealth_today_s_power:,.0f})\n"
            f"Market Index: {selected_market_index}\n"
            f"Annual Inflation: {inflation_rate:.1%}\n"
            f"Initial Monthly Expenses: €{inital_monthly_expenses_total:,.0f}\n"
        )
        text_plot_2 = (f"Annual Inflation: {inflation_rate:.1%}\n"
            f"Initial Monthly Expenses: €{inital_monthly_expenses_total:,.0f}\n")
        from flask import session
        session['simulation_results'] = {
            't': t.tolist(), # Convert numpy array to list
            'median_total': median_total.tolist(),
            'df_data': df.to_dict('records'), # Convert DataFrame to list of dicts
            'p5_total': p5_total.tolist(),
            'p80_total': p80_total.tolist(),
            'prob_below_threshold': prob_below_threshold.tolist(),
            'prob_zero_cash': prob_zero_cash.tolist(),
            'cum_income': cum_income.tolist(),
            'cum_expense': cum_expense.tolist(),
            'cum_net': cum_net.tolist(),
            'cum_first_person': cum_first_person.tolist(),
            'cum_second_person': cum_second_person.tolist(),
            'median_cash': median_cash.tolist(),
            'median_port': median_port.tolist(),
            'cash_paths': cash_paths.tolist(),
            'port_paths': port_paths.tolist(),
            'total_paths': total_paths.tolist(),
            'cum_large_payments': cum_large_payments.tolist(),
            'start_date_str': start_date_str, # Store original parameters
            'end_date_str': end_date_str,
            'initial_cash': initial_cash,
            'invest_frac': invest_frac,
            'threshold_cash': threshold_cash,
            'selected_market_index': selected_market_index,
            'inflation_rate': inflation_rate,
            'initial_monthly_expenses': initial_monthly_expenses, # Add other input parameters if needed for summary/text
            'median_final_wealth': median_final_wealth,
            'median_final_wealth_after_taxes': median_final_wealth_after_taxes,
            'median_final_wealth_today_s_power': median_final_wealth_today_s_power,
            'median_final_wealth_today_s_power_before_taxes': median_final_wealth_today_s_power_before_taxes,
            'prob_zero_cash_final': prob_zero_cash_final,
            'percentile_90_wealth': percentile_90_wealth,
            'simulation_summary_html': simulation_summary, # Store the HTML summary directly
            'plot_params_text': plot_params_text,
            'text_plot_2_params': text_plot_2,
            'initial_portfolio_value': initial_portfolio_value, # Also store calculated values used in text
        }

        plots = {}

        # --- Plot 1: Key Financial Trajectories with 5–80%ile Shading ---
        #fig1, ax1 = plt.subplots(figsize=(10, 6))
        t_years = t / 12.0
        
        
        
        
        fig1 = plt.figure(figsize=(10, 10)) # Increase figure height to accommodate text
        gs1 = gridspec.GridSpec(2, 1, height_ratios=[4, 1]) # 4 parts for plot, 1 for text
        ax1 = fig1.add_subplot(gs1[0, 0]) # Main plot
        ax1_text = fig1.add_subplot(gs1[1, 0]) # Text subplot
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
        ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x)}y'))
        ax1.legend(loc='upper left', fontsize=9, frameon=True, shadow=True, fancybox=True)
        ax1_text.text(0.0, 1.0, plot_params_text, transform=ax1_text.transAxes,
                 fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.6))
        ax1_text.axis('off') # Hide the axes for the text subplot
        fig1.tight_layout(rect=[0, 0.03, 1, 1]) # Adjust layout to prevent o
        plots['plot1'] = plot_to_base64(fig1)
        '''
        # --- Plot 2: Cumulative Cash Components Over Time ---
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.plot(t, cum_first_person, label='Cumulative First Person Income', linestyle=':', marker='o', markersize=3, color='darkgreen')
        ax2.plot(t, cum_second_person, label='Cumulative Second Person Income', linestyle='--', marker='s', markersize=3, color='darkblue')
        ax2.plot(t, cum_expense, label='Cumulative Expenses', linestyle='-', color='firebrick')
        #ax2.plot(t, cum_large_payments, label='Cumulative Large Payments/Withdrawals', linestyle='-.', marker='x', markersize=4, color='gray')
        ax2.plot(t, cum_income, label = 'Cumulative Total Income', linestyle='-', linewidth=1.5, color='teal')
        ax2.plot(t, cum_net, label= 'Cumulative Net Cash', linewidth = 3.5, color='black', linestyle='-')
        ax2.axhline(threshold_cash, color='red', linestyle='--', linewidth=1.5, label=f'Cash Threshold €{threshold_cash:,.0f}')
        ax2.set_xlabel('Months since Start Date')
        ax2.set_ylabel('Amount (€)')
        ax2.set_title('Cumulative Cash Components Over Time')
        ax2.legend(loc='upper left', fontsize=9, frameon=True, shadow=True, fancybox=True)
        apply_scientific_style(ax2, plot_params_text)
        plots['plot2'] = plot_to_base64(fig2)
        '''
        # --- Plot 2: Cumulative Cash Components Over Time ---
        fig2 = plt.figure(figsize=(10, 10)) # Increase figure height for text
        gs2 = gridspec.GridSpec(2, 1, height_ratios=[4, 1]) # 4 parts for plot, 1 for text
        ax2 = fig2.add_subplot(gs2[0, 0]) # Main plot
        ax2_text = fig2.add_subplot(gs2[1, 0]) # Text subplot

        marker_interval = 6 
        ax2.plot(t[::marker_interval], cum_first_person[::marker_interval], label='Cumulative First Person Income', linestyle=':', marker='o', markersize=4, color='darkgreen', markevery=marker_interval)
        ax2.plot(t[::marker_interval], cum_second_person[::marker_interval], label='Cumulative Second Person Income', linestyle='--', marker='s', markersize=4, color='darkblue', markevery=marker_interval)
        ax2.plot(t, cum_expense, label='Cumulative Expenses', linestyle='-', color='firebrick', linewidth=1.5) # No markers, or very subtle if needed
        # ax2.plot(t[::marker_interval], cum_large_payments[::marker_interval], label='Cumulative Large Payments/Withdrawals', linestyle='-.', marker='x', markersize=4, color='gray', markevery=marker_interval) # If you decide to re-include this
        ax2.plot(t, cum_income, label='Cumulative Total Income', linestyle='-', linewidth=2, color='teal') # Increased linewidth
        ax2.plot(t, cum_net, label='Cumulative Net Cash', linewidth=3.5, color='black', linestyle='-')

        ax2.axhline(threshold_cash, color='red', linestyle='--', linewidth=1.5, label=f'Cash Threshold €{threshold_cash:,.0f}')
        ax2.set_xlabel('Months since Start Date')
        ax2.set_ylabel('Amount (€)')
        ax2.set_title('Cumulative Cash Components Over Time')

        # Position the legend to avoid overlapping with data if possible. 'upper left' is usually good for cumulative plots.
        ax2.legend(loc='upper left', fontsize=9, frameon=True, shadow=True, fancybox=True)

        apply_scientific_style(ax2) # Call without plot_params_text as it's handled separately

        # Add text to the dedicated subplot
        
        ax2_text.text(0.0, 1.0, text_plot_2, transform=ax2_text.transAxes,
                    fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.6))
        ax2_text.axis('off') # Hide the axes for the text subplot
        fig2.tight_layout(rect=[0, 0.03, 1, 1]) # Adjust layout
        plots['plot2'] = plot_to_base64(fig2)
        
        
        
        
        
        # --- Plot 3: Monthly Cash Flow Components Over Time ---
        fig3, ax3 = plt.subplots(figsize=(10, 10))
        ax3.plot(df.index, df['expenses'], label='Total Monthly Expenses', linestyle='dashed', color='red', marker='v', markersize=2, alpha=0.7)
        ax3.plot(df.index, df['income'], label='Total Monthly Income', linestyle='dashed', color='green', marker='^', markersize=2, alpha=0.7)
        ax3.plot(df.index, df['net_cash'], label='Monthly Net Cash', linestyle='solid', color='blue', linewidth=1.5)
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Amount (€ / month)')
        ax3.set_title('Monthly Cash Flow Components Over Time')
        apply_scientific_style(ax3, plot_params_text)
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax3.legend(loc='upper left', fontsize=9, frameon=True, shadow=True, fancybox=True)
        fig3.autofmt_xdate()
        plots['plot3'] = plot_to_base64(fig3)

        # --- Plot 4: Histogram of Final Total Wealth ---
        fig4, ax4 = plt.subplots(figsize=(10, 10))
        final = total_paths[:, -1]
        ax4.hist(final, bins=30, density=True, alpha=0.8, color='skyblue', edgecolor='black')
        ax4.axvline(median_final_wealth, color='red', linestyle='--', label=f'Median Final Wealth: €{median_final_wealth:,.0f}')
        ax4.axvline(percentile_5, color = 'black', label= f'5th Percentile: €{percentile_5:,.0f}', linestyle='--')
        ax4.set_xlabel('Final Total Wealth (€)')
        ax4.set_ylabel('Probability Density')
        ax4.set_title(f'Histogram of Final Total Wealth ({df.index[-1].strftime("%b %Y")})')
        ax4.legend(loc='upper right', fontsize=9, frameon=True, shadow=True, fancybox=True)
        apply_scientific_style(ax4, plot_params_text)
        plots['plot4'] = plot_to_base64(fig4)

        # --- Plot 5: Probability of Cash Account Reaching Critical Levels Over Time ---
        fig5, ax5 = plt.subplots(figsize=(10, 10))
        ax5.plot(t, prob_below_threshold, label=f'Probability Cash < €{threshold_cash:,.0f}', color='blue', linewidth=2, marker='.')
        ax5.plot(t, prob_zero_cash, label='Probability Cash <= €0', color='red', linestyle='--', linewidth=2, marker='x')
        ax5.set_xlabel('Months since Start Date')
        ax5.set_ylabel('Probability')
        ax5.set_title('Probability of Cash Account Reaching Critical Levels Over Time')
        ax5.set_ylim(0, 1)
        ax5.legend(loc='upper right', fontsize=9, frameon=True, shadow=True, fancybox=True)
        apply_scientific_style(ax5, plot_params_text)
        plots['plot5'] = plot_to_base64(fig5) # This was plot5 in the original, but results.html refers to it as plot3, so I will map it accordingly

        # --- Plot 6: Median Total Wealth Breakdown: Cash vs. Portfolio (Stacked Area Plot) ---
        fig6, ax6 = plt.subplots(figsize=( 10, 10))
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
        fig7, ax7 = plt.subplots(figsize=(10, 10))
        df_annual = df.resample('Y').sum(numeric_only=True)
        annual_net_cash = df_annual['net_cash']
        annual_labels = [str(x.year) for x in df_annual.index]
        bars = ax7.bar(annual_labels, annual_net_cash, color=['skyblue' if x >= 0 else 'lightcoral' for x in annual_net_cash], edgecolor='black', linewidth=0.7)
        ax7.axhline(0, color='red', linestyle='--', linewidth=1.5)
        ax7.set_xlabel('Year')
        ax7.set_ylabel('Annual Net Cash Flow (€)')
        ax7.set_title('Annual Net Cash Flow (Income - Expenses + Large Payments)')
        ax7.tick_params(axis='x', rotation=45) # Removed ha='right'
        ax7.legend(loc='upper right', fontsize=9, frameon=True, shadow=True, fancybox=True) 
        apply_scientific_style(ax7, plot_params_text)
        for bar in bars:
            yval = bar.get_height()
            # Adjust y-position slightly to move text just above/below the bar
            text_y_offset = 5000 # Smaller offset for closer placement
            ha = 'center'
            va = 'bottom' if yval >= 0 else 'top'
            
            
            rotation_angle = 90 # Rotate text by 90 degrees
            
            ax7.text(bar.get_x() + bar.get_width()/2, yval + (text_y_offset if yval >= 0 else -text_y_offset), f'€{yval:,.0f}',
                     ha=ha, va=va, fontsize=8, color='black', rotation=rotation_angle) # Add rotation
            
        plots['plot7'] = plot_to_base64(fig7)

        fig8, ax8 = plt.subplots(figsize=(10, 6))
        final_wealth_values = total_paths[:, -1]
        sorted_final_wealth = np.sort(final_wealth_values)
        cdf = np.arange(1, len(sorted_final_wealth) + 1) / len(sorted_final_wealth)

        ax8.plot(sorted_final_wealth, cdf, color='darkgreen', linewidth=2)
        ax8.set_xlabel('Final Total Wealth (€)')
        ax8.set_ylabel('Cumulative Probability')
        ax8.set_title(f'Cumulative Distribution of Final Total Wealth ({df.index[-1].strftime("%b %Y")})')
        ax8.grid(True, which='major', linestyle='-', linewidth='0.7', color='lightgray', alpha=0.8)
        ax8.grid(True, which='minor', linestyle=':', linewidth='0.5', color='lightgray', alpha=0.5)
        ax8.set_ylim(0, 1) # CDF always ranges from 0 to 1

        # Add lines for median and 5th percentile for better context
        ax8.axvline(median_final_wealth, color='red', linestyle='--', label=f'Median: €{median_final_wealth:,.0f}')
        ax8.axvline(percentile_5, color='orange', linestyle='--', label=f'5th Percentile: €{percentile_5:,.0f}')
        ax8.axvline(percentile_90_wealth, color='blue', linestyle='--', label=f'90th Percentile: €{percentile_90_wealth:,.0f}') # Assuming percentile_90_wealth is actually the 10th percentile
        ax8.legend(loc='upper left', fontsize=9, frameon=True, shadow=True, fancybox=True)

        apply_scientific_style(ax8) # Apply the consistent style
        plots['plot8'] = plot_to_base64(fig8) # Assign to 'plot8'
        #return render_template('results.html', plots=plots, final_wealth=f"€{median_final_wealth:,.0f}", final_wealth_after_tax = f"€{median_final_wealth_after_taxes:,.0f}")
        return render_template(
            'results.html', 
            plots=plots, 
            final_wealth=f"€{median_final_wealth:,.0f}", 
            final_wealth_after_tax=f"€{median_final_wealth_after_taxes:,.0f}",
            simulation_summary=simulation_summary # Pass the new summary
        )
    
    
    except ValueError as e:
        # Handle invalid input errors
        return render_template('simulator_parameters.html', error=f"Input Error: {e}")
    except Exception as e:
        # Handle any other unexpected errors
        return render_template('simulator_parameters.html', error=f"An unexpected error occurred: {e}")
from flask import session # Make sure this is imported at the top

@app.route('/simulator/download_report', methods=['GET'])
def download_report():
    """
    Generates and serves the PDF report based on simulation parameters stored in the session.
    """
    from flask import session
    session_results = session.get('simulation_results')

    if not session_results:
        # User tried to access download without running simulation or session expired
        # Redirect them back to the simulator or an error page
        return redirect(url_for('simulator_index', error="Please run a simulation first to generate a report."))

    try:
        # Reconstruct data from session and convert back to numpy/pandas where needed
        t = np.array(session_results['t'])
        median_total = np.array(session_results['median_total'])

        df = pd.DataFrame(session_results['df_data'])
        # IMPORTANT: Reconstruct the datetime index for the DataFrame
        df.index = pd.to_datetime(session_results['df_index_dates'])


        p5_total = np.array(session_results['p5_total'])
        p80_total = np.array(session_results['p80_total'])
        prob_below_threshold = np.array(session_results['prob_below_threshold'])
        prob_zero_cash = np.array(session_results['prob_zero_cash'])
        cum_income = np.array(session_results['cum_income'])
        cum_expense = np.array(session_results['cum_expense'])
        cum_net = np.array(session_results['cum_net'])
        cum_first_person = np.array(session_results['cum_first_person'])
        cum_second_person = np.array(session_results['cum_second_person'])
        median_cash = np.array(session_results['median_cash'])
        median_port = np.array(session_results['median_port'])
        total_paths = np.array(session_results['total_paths'])
        cum_large_payments = np.array(session_results['cum_large_payments'])

        # Retrieve direct values for summary and text
        median_final_wealth = session_results['median_final_wealth']
        median_final_wealth_after_taxes = session_results['median_final_wealth_after_taxes']
        median_final_wealth_today_s_power = session_results['median_final_wealth_today_s_power']
        median_final_wealth_today_s_power_before_taxes = session_results['median_final_wealth_today_s_power_before_taxes']
        prob_zero_cash_final = session_results['prob_zero_cash_final']
        percentile_90_wealth = session_results['percentile_90_wealth']
        simulation_summary_text = session_results['simulation_summary_html'] # Use the pre-generated HTML summary
        plot_params_text = session_results['plot_params_text']
        text_plot_2_params = session_results['text_plot_2_params']
        threshold_cash = session_results['threshold_cash']
        percentile_5 = session_results['percentile_5'] # Ensure this is stored in session
        start_date_str = session_results['start_date_str'] # Needed for date recreation
        end_date_str = session_results['end_date_str'] # Needed for plot titles


        # --- Re-generate Figures from retrieved data for PDF generation ---
        figures_to_pdf = {}

        # Plot 1
        fig1 = plt.figure(figsize=(10, 10))
        gs1 = gridspec.GridSpec(2, 1, height_ratios=[4, 1])
        ax1 = fig1.add_subplot(gs1[0, 0])
        ax1_text = fig1.add_subplot(gs1[1, 0])
        t_years = t / 12.0
        ax1.plot(t_years, median_cash, label='Median Cash Account', linestyle='--', marker='^', markersize=4, color='green')
        ax1.plot(t_years, median_port, label='Median Portfolio Value', linestyle='-', marker='D', markersize=3, color='purple')
        ax1.fill_between(t_years, p5_total, p80_total, color='orange', alpha=0.3, label='5–80th %ile Total Wealth')
        ax1.plot(t_years, median_total, label='Median Total Wealth', linewidth=2.5, color='orange', linestyle='-')
        ax1.axhline(threshold_cash, color='crimson', linestyle='--', linewidth=1.5, label=f'Cash Threshold €{threshold_cash:,.0f}')
        ax1.set_xlabel('Years since Start Date')
        ax1.set_ylabel('Amount (€)')
        ax1.set_title('Key Financial Trajectories with 5–80%ile Shading')
        apply_scientific_style(ax1)
        ax1.xaxis.set_major_locator(ticker.MultipleLocator(5))
        ax1.xaxis.set_minor_locator(ticker.MultipleLocator(1))
        ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x)}y'))
        ax1.legend(loc='upper left', fontsize=9, frameon=True, shadow=True, fancybox=True)
        ax1_text.text(0.0, 1.0, plot_params_text, transform=ax1_text.transAxes,
                 fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.6))
        ax1_text.axis('off')
        fig1.tight_layout(rect=[0, 0.03, 1, 1])
        figures_to_pdf['plot1'] = fig1


        # Plot 2
        fig2 = plt.figure(figsize=(10, 10))
        gs2 = gridspec.GridSpec(2, 1, height_ratios=[4, 1])
        ax2 = fig2.add_subplot(gs2[0, 0])
        ax2_text = fig2.add_subplot(gs2[1, 0])
        marker_interval = 6
        ax2.plot(t[::marker_interval], cum_first_person[::marker_interval], label='Cumulative First Person Income', linestyle=':', marker='o', markersize=4, color='darkgreen', markevery=marker_interval)
        ax2.plot(t[::marker_interval], cum_second_person[::marker_interval], label='Cumulative Second Person Income', linestyle='--', marker='s', markersize=4, color='darkblue', markevery=marker_interval)
        ax2.plot(t, cum_expense, label='Cumulative Expenses', linestyle='-', color='firebrick', linewidth=1.5)
        ax2.plot(t, cum_income, label='Cumulative Total Income', linestyle='-', linewidth=2, color='teal')
        ax2.plot(t, cum_net, label='Cumulative Net Cash', linewidth=3.5, color='black', linestyle='-')
        ax2.axhline(threshold_cash, color='red', linestyle='--', linewidth=1.5, label=f'Cash Threshold €{threshold_cash:,.0f}')
        ax2.set_xlabel('Months since Start Date')
        ax2.set_ylabel('Amount (€)')
        ax2.set_title('Cumulative Cash Components Over Time')
        ax2.legend(loc='upper left', fontsize=9, frameon=True, shadow=True, fancybox=True)
        apply_scientific_style(ax2)
        ax2_text.text(0.0, 1.0, text_plot_2_params, transform=ax2_text.transAxes,
                    fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.6))
        ax2_text.axis('off')
        fig2.tight_layout(rect=[0, 0.03, 1, 1])
        figures_to_pdf['plot2'] = fig2

        # Plot 3
        fig3_obj, ax3 = plt.subplots(figsize=(10, 6))
        ax3.plot(df.index, df['expenses'], label='Total Monthly Expenses', linestyle='dashed', color='red', marker='v', markersize=2, alpha=0.7)
        ax3.plot(df.index, df['income'], label='Total Monthly Income', linestyle='dashed', color='green', marker='^', markersize=2, alpha=0.7)
        ax3.plot(df.index, df['net_cash'], label='Monthly Net Cash', linestyle='solid', color='blue', linewidth=1.5)
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Amount (€ / month)')
        ax3.set_title('Monthly Cash Flow Components Over Time')
        apply_scientific_style(ax3)
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax3.legend(loc='upper left', fontsize=9, frameon=True, shadow=True, fancybox=True)
        fig3_obj.autofmt_xdate()
        figures_to_pdf['plot3'] = fig3_obj

        # Plot 4
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        final = total_paths[:, -1]
        ax4.hist(final, bins=30, density=True, alpha=0.8, color='skyblue', edgecolor='black')
        ax4.axvline(median_final_wealth, color='red', linestyle='--', label=f'Median Final Wealth: €{median_final_wealth:,.0f}')
        ax4.axvline(percentile_5, color = 'black', label= f'5th Percentile: €{percentile_5:,.0f}', linestyle='--')
        ax4.set_xlabel('Final Total Wealth (€)')
        ax4.set_ylabel('Probability Density')
        end_date_for_title = datetime.strptime(end_date_str, '%Y-%m-%d').strftime('%b %Y')
        ax4.set_title(f'Histogram of Final Total Wealth ({end_date_for_title})')
        ax4.legend(loc='upper right', fontsize=9, frameon=True, shadow=True, fancybox=True)
        apply_scientific_style(ax4)
        figures_to_pdf['plot4'] = fig4

        # Plot 5
        fig5, ax5 = plt.subplots(figsize=(10, 6))
        ax5.plot(t, prob_below_threshold, label=f'Probability Cash < €{threshold_cash:,.0f}', color='blue', linewidth=2, marker='.')
        ax5.plot(t, prob_zero_cash, label='Probability Cash <= €0', color='red', linestyle='--', linewidth=2, marker='x')
        ax5.set_xlabel('Months since Start Date')
        ax5.set_ylabel('Probability')
        ax5.set_title('Probability of Cash Account Reaching Critical Levels Over Time')
        ax5.set_ylim(0, 1)
        ax5.legend(loc='upper right', fontsize=9, frameon=True, shadow=True, fancybox=True)
        apply_scientific_style(ax5)
        figures_to_pdf['plot5'] = fig5

        # Plot 6
        fig6, ax6 = plt.subplots(figsize=( 10, 6))
        ax6.stackplot(t, median_cash, median_port, labels=['Median Cash Account', 'Median Portfolio Value'], alpha=0.8, colors=['lightgreen', 'lightblue'])
        ax6.plot(t, median_total, color='black', linestyle='--', linewidth=2, label='Median Total Wealth')
        ax6.set_xlabel('Months since Start Date')
        ax6.set_ylabel('Amount (€)')
        ax6.set_title('Median Total Wealth Breakdown: Cash vs. Portfolio')
        ax6.legend(loc='upper left', fontsize=9, frameon=True, shadow=True, fancybox=True)
        apply_scientific_style(ax6)
        figures_to_pdf['plot6'] = fig6

        # Plot 7
        fig7, ax7 = plt.subplots(figsize=(10, 6))
        df_annual = df.resample('Y').sum(numeric_only=True) # Use the correctly reconstructed df
        annual_net_cash = df_annual['net_cash']
        annual_labels = [str(x.year) for x in df_annual.index]
        bars = ax7.bar(annual_labels, annual_net_cash, color=['skyblue' if x >= 0 else 'lightcoral' for x in annual_net_cash], edgecolor='black', linewidth=0.7)
        ax7.axhline(0, color='red', linestyle='--', linewidth=1.5)
        ax7.set_xlabel('Year')
        ax7.set_ylabel('Annual Net Cash Flow (€)')
        ax7.set_title('Annual Net Cash Flow (Income - Expenses + Large Payments)')
        ax7.tick_params(axis='x', rotation=45)
        apply_scientific_style(ax7)
        for bar in bars:
            yval = bar.get_height()
            text_y_offset = 5000
            ha = 'center'
            va = 'bottom' if yval >= 0 else 'top'
            rotation_angle = 90
            ax7.text(bar.get_x() + bar.get_width()/2, yval + (text_y_offset if yval >= 0 else -text_y_offset), f'€{yval:,.0f}',
                     ha=ha, va=va, fontsize=8, color='black', rotation=rotation_angle)
        figures_to_pdf['plot7'] = fig7

        # Plot 8 (CDF)
        fig8, ax8 = plt.subplots(figsize=(10, 6))
        final_wealth_values = total_paths[:, -1]
        sorted_final_wealth = np.sort(final_wealth_values)
        cdf = np.arange(1, len(sorted_final_wealth) + 1) / len(sorted_final_wealth)

        ax8.plot(sorted_final_wealth, cdf, color='darkgreen', linewidth=2)
        ax8.set_xlabel('Final Total Wealth (€)')
        ax8.set_ylabel('Cumulative Probability')
        ax8.set_title(f'Cumulative Distribution of Final Total Wealth ({end_date_for_title})')
        ax8.grid(True, which='major', linestyle='-', linewidth='0.7', color='lightgray', alpha=0.8)
        ax8.grid(True, which='minor', linestyle=':', linewidth='0.5', color='lightgray', alpha=0.5)
        ax8.set_ylim(0, 1)

        ax8.axvline(median_final_wealth, color='red', linestyle='--', label=f'Median: €{median_final_wealth:,.0f}')
        ax8.axvline(percentile_5, color='orange', linestyle='--', label=f'5th Percentile: €{percentile_5:,.0f}')
        ax8.axvline(percentile_90_wealth, color='blue', linestyle='--', label=f'90th Percentile: €{percentile_90_wealth:,.0f}')
        ax8.legend(loc='upper left', fontsize=9, frameon=True, shadow=True, fancybox=True)

        apply_scientific_style(ax8)
        figures_to_pdf['plot8'] = fig8


        # Generate the PDF report
        pdf_file_path = "/tmp/simulation_report.pdf" # Use /tmp for Cloud Run (writable temp directory)
        generate_pdf_report(figures_to_pdf, simulation_summary_text, pdf_file_path)

        # IMPORTANT: Only clear session if you're certain it's the final action
        # For now, let's remove it to allow subsequent downloads or viewing results page again
        # session.pop('simulation_results', None) # Comment this out for now for easier testing

        # Send the generated PDF file
        return send_file(pdf_file_path, as_attachment=True, download_name="financial_simulation_report.pdf", mimetype='application/pdf')

    except Exception as e:
        import traceback
        # Return a plain text error, not HTML code, when PDF generation fails.
        return f"Error generating PDF from session data: {e}<br><pre>{traceback.format_exc()}</pre>", 500

if __name__ == '__main__':
    # This block is for local development. In a Canvas environment, the app is run differently.
    # To run locally, create a 'templates' folder and put index.html and results.html inside.
    # Create a 'static' folder and put placeholder images (or your actual images) inside.
    # You might need to install Flask: pip install Flask
    # Then run: python app.py
    app.run(debug=True) # For local debugging


'''
        # --- Plot 8: Cumulative Impact of Large Financial Events ---
        fig8, ax8 = plt.subplots(figsize=(10, 6))
        ax8.plot(t, cum_large_payments, label='Cumulative Large Payments/Withdrawals', color='purple', linewidth=2, marker='.', markersize=4)
        ax8.set_xlabel('Months since Start Date')
        ax8.set_ylabel('Amount (€)')
        ax8.set_title('Cumulative Impact of Large Financial Events')
        ax8.legend(loc='upper left', fontsize=9, frameon=True, shadow=True, fancybox=True)
        apply_scientific_style(ax8, plot_params_text)
        plots['plot8'] = plot_to_base64(fig8)
        '''