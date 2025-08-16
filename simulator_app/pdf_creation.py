from flask import session # Make sure this is imported at the top
import io
import base64
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.gridspec as gridspec # Add this import at the top of main_app.py

from simulator_app.simulation_v2_inflation import run_simulation

import matplotlib
matplotlib.use('Agg') # Use the 'Agg' backend for non-interactive plotting
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
from main_app.py import apply_scientific_style

def download_report():
    """
    Generates and serves the PDF report based on simulation parameters stored in the session.
    """
    
    session_results = session.get('simulation_results')

    if not session_results:
        # Handle case where no simulation results are in session (e.g., user refreshed/came directly)
        return "No simulation results found in session. Please run a simulation first.", 400

    try:
        # Retrieve data from session and convert back to numpy/pandas where needed
        t = np.array(session_results['t'])
        median_total = np.array(session_results['median_total'])
        df = pd.DataFrame(session_results['df_data'])
        # Ensure df has a datetime index if needed for plotting (e.g., for plot 3 or 7)
        # Assuming original df index was dates, convert back if necessary
        # Example: if your original df had a 'date' column as index, you'd do:
        # df.index = pd.to_datetime([d['index'] for d in session_results['df_data']]) if 'index' in session_results['df_data'][0] else df.index

        # If df.index was datetime, ensure it's re-parsed as such for resample/plotting
        df.index = pd.to_datetime(df['date']) # Assuming 'date' column was created when converting to dicts
                                             # You might need to adjust this based on how df.to_dict('records') stores index

        # If original df.index was derived from a start_date, you can reconstruct
        start_date_str = session_results['start_date_str']
        end_date_str = session_results['end_date_str'] # Not strictly needed here but good to have
        inflation_rate = session_results['inflation_rate']
        initial_monthly_expenses = session_results['initial_monthly_expenses'] # Retrieve this

        # Reconstruct date index for plotting if necessary (especially for df.index)
        # This is critical if your plots use df.index (like Plot 3 and Plot 7)
        if 'start_date_str' in session_results:
            # Assuming t represents months since start_date_str
            start_date_dt = datetime.strptime(session_results['start_date_str'], '%Y-%m-%d')
            # Create a monthly date range for plotting x-axes
            # For Plot 3 and Plot 7, you might need a proper date range.
            # If df.index itself was stored in df_data, use that.
            # Otherwise, regenerate like this:
            all_dates = pd.date_range(start=start_date_dt, periods=len(t), freq='MS')
            # You might need to re-assign this to df if df relies on it
            # For this example, let's assume `df` coming from session_results['df_data'] already has a 'date' column
            # that can be converted to index. If not, you may need to map `all_dates` to `df.index`

        # Reconstruct other numpy arrays
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
        cum_large_payments = np.array(session_results['cum_large_payments']) # Ensure this is always a numpy array

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
        threshold_cash = session_results['threshold_cash'] # Need this for the plot title

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
        fig2 = plt.figure(figsize=(10, 10)) # Increase figure height for text
        gs2 = gridspec.GridSpec(2, 1, height_ratios=[4, 1]) # 4 parts for plot, 1 for text
        ax2 = fig2.add_subplot(gs2[0, 0]) # Main plot
        ax2_text = fig2.add_subplot(gs2[1, 0]) # Text subplot

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
        dates_for_plot3_and_7 = pd.date_range(start=start_date_str, periods=len(t), freq='MS') # Reconstruct date index
        ax3.plot(dates_for_plot3_and_7, df['expenses'], label='Total Monthly Expenses', linestyle='dashed', color='red', marker='v', markersize=2, alpha=0.7)
        ax3.plot(dates_for_plot3_and_7, df['income'], label='Total Monthly Income', linestyle='dashed', color='green', marker='^', markersize=2, alpha=0.7)
        ax3.plot(dates_for_plot3_and_7, df['net_cash'], label='Monthly Net Cash', linestyle='solid', color='blue', linewidth=1.5)
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
        # df.index[-1] might not be available directly if df's index isn't properly handled on retrieval
        # Better to use end_date_str or derived date
        end_date_for_title = datetime.strptime(session_results['end_date_str'], '%Y-%m-%d').strftime('%b %Y')
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
        # Need to reconstruct df_annual for this plot
        # Recreate df from session data and set index as dates for resample
        df_reconstructed = pd.DataFrame(session_results['df_data'])
        df_reconstructed['date'] = pd.to_datetime(df_reconstructed['date']) # Ensure 'date' column is datetime
        df_reconstructed = df_reconstructed.set_index('date') # Set it as index

        df_annual = df_reconstructed.resample('Y').sum(numeric_only=True)
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
        ax8.set_title(f'Cumulative Distribution of Final Total Wealth ({end_date_for_title})') # Using end_date_for_title here
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

        # Clear session results after PDF generation to manage session size
        session.pop('simulation_results', None)

        # Send the generated PDF file
        return send_file(pdf_file_path, as_attachment=True, download_name="financial_simulation_report.pdf", mimetype='application/pdf')

    except Exception as e:
        # Provide a more informative error for debugging
        import traceback
        return f"Error generating PDF from session data: {e}<br><pre>{traceback.format_exc()}</pre>", 500