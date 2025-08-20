import io
import base64
import matplotlib
matplotlib.use('Agg') # Use the 'Agg' backend for non-interactive plotting
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from datetime import datetime


def plot_to_base64(fig):
    """Converts a matplotlib figure to a base64 encoded PNG image."""
    buf = io.BytesIO()
    # Increased pad_inches slightly to give more room around the plot
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.8) # Changed from 0.5 to 0.8
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig) # Close the figure to free up memory
    return img_base64

def apply_scientific_style(ax):
    """Applies a consistent scientific style to a matplotlib axes object."""
    ax.tick_params(axis='both', which='major', labelsize=10, width=1.5, length=6, direction='inout')
    ax.tick_params(axis='both', which='minor', labelsize=8, width=0.75, length=3, direction='inout')
    ax.minorticks_on()
    ax.grid(True, which='major', linestyle='-', linewidth='0.7', color='lightgray', alpha=0.8)
    ax.grid(True, which='minor', linestyle=':', linewidth='0.5', color='lightgray', alpha=0.5)
    ax.set_title(ax.get_title(), fontsize=14, weight='bold')
    ax.set_xlabel(ax.get_xlabel(), fontsize=12)
    ax.set_ylabel(ax.get_ylabel(), fontsize=12)

    if ax.get_legend() is not None:
        ax.legend(fontsize=9, frameon=True, shadow=True, fancybox=True)
    else:
        pass
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def generate_plot1(t, median_cash, median_port, p5_total, p80_total, median_total, threshold_cash, plot_params_text):
    fig = plt.figure(figsize=(12, 10))
    gs1 = gridspec.GridSpec(2, 1, height_ratios=[4, 1])
    ax1 = fig.add_subplot(gs1[0, 0])
    ax1_text = fig.add_subplot(gs1[1, 0])
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
    fig.tight_layout(rect=[0, 0.03, 1, 1]) # Already has tight_layout
    return fig

def generate_plot2(t, cum_first_person, cum_second_person, cum_expense, cum_income, cum_net, threshold_cash, text_plot_2_params):
    fig = plt.figure(figsize=(12, 10))
    gs2 = gridspec.GridSpec(2, 1, height_ratios=[4, 1])
    ax2 = fig.add_subplot(gs2[0, 0])
    ax2_text = fig.add_subplot(gs2[1, 0])
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
    fig.tight_layout(rect=[0, 0.03, 1, 1]) # Already has tight_layout
    return fig

def generate_plot3(df):
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.plot(df.index, df['expenses'], label='Total Monthly Expenses', linestyle='dashed', color='red', marker='v', markersize=2, alpha=0.7)
    ax.plot(df.index, df['income'], label='Total Monthly Income', linestyle='dashed', color='green', marker='^', markersize=2, alpha=0.7)
    ax.plot(df.index, df['net_cash'], label='Monthly Net Cash', linestyle='solid', color='blue', linewidth=1.5)
    ax.set_xlabel('Date')
    ax.set_ylabel('Amount (€ / month)')
    ax.set_title('Monthly Cash Flow Components Over Time')
    apply_scientific_style(ax)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.legend(loc='upper left', fontsize=9, frameon=True, shadow=True, fancybox=True)
    fig.autofmt_xdate()
    fig.tight_layout() # Added tight_layout
    return fig

def generate_plot4(total_paths, median_final_wealth, percentile_5, end_date_str):
    fig, ax = plt.subplots(figsize=(12, 10))
    final = total_paths[:, -1]
    ax.hist(final, bins=30, density=True, alpha=0.8, color='skyblue', edgecolor='black')
    ax.axvline(median_final_wealth, color='red', linestyle='--', label=f'Median Final Wealth: €{median_final_wealth:,.0f}')
    ax.axvline(percentile_5, color = 'black', label= f'5th Percentile: €{percentile_5:,.0f}', linestyle='--')
    ax.set_xlabel('Final Total Wealth (€)')
    ax.set_ylabel('Probability Density')
    end_date_for_title = datetime.strptime(end_date_str, '%Y-%m-%d').strftime('%b %Y')
    ax.set_title(f'Histogram of Final Total Wealth ({end_date_for_title})')
    ax.legend(loc='upper right', fontsize=9, frameon=True, shadow=True, fancybox=True)
    apply_scientific_style(ax)
    fig.tight_layout() # Added tight_layout
    return fig

def generate_plot5(t, prob_below_threshold, prob_zero_cash, threshold_cash):
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.plot(t, prob_below_threshold, label=f'Probability Cash < €{threshold_cash:,.0f}', color='blue', linewidth=2, marker='.')
    ax.plot(t, prob_zero_cash, label='Probability Cash <= €0', color='red', linestyle='--', linewidth=2, marker='x')
    ax.set_xlabel('Months since Start Date')
    ax.set_ylabel('Probability')
    ax.set_title('Probability of Cash Account Reaching Critical Levels Over Time')
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', fontsize=9, frameon=True, shadow=True, fancybox=True)
    apply_scientific_style(ax)
    fig.tight_layout() # Added tight_layout
    return fig

def generate_plot6(t, median_cash, median_port, median_total):
    fig, ax = plt.subplots(figsize=( 12, 10))
    ax.stackplot(t, median_cash, median_port, labels=['Median Cash Account', 'Median Portfolio Value'], alpha=0.8, colors=['lightgreen', 'lightblue'])
    ax.plot(t, median_total, color='black', linestyle='--', linewidth=2, label='Median Total Wealth')
    ax.set_xlabel('Months since Start Date')
    ax.set_ylabel('Amount (€)')
    ax.set_title('Median Total Wealth Breakdown: Cash vs. Portfolio')
    ax.legend(loc='upper left', fontsize=9, frameon=True, shadow=True, fancybox=True)
    apply_scientific_style(ax)
    fig.tight_layout() # Added tight_layout
    return fig

def generate_plot7(df):
    fig, ax = plt.subplots(figsize=(12, 10))
    df_annual = df.resample('Y').sum(numeric_only=True)
    annual_net_cash = df_annual['net_cash']
    annual_labels = [str(x.year) for x in df_annual.index]
    bars = ax.bar(annual_labels, annual_net_cash, color=['skyblue' if x >= 0 else 'lightcoral' for x in annual_net_cash], edgecolor='black', linewidth=0.7)
    ax.axhline(0, color='red', linestyle='--', linewidth=1.5)
    ax.set_xlabel('Year')
    ax.set_ylabel('Annual Net Cash Flow (€)')
    ax.set_title('Annual Net Cash Flow (Income - Expenses + Large Payments)')
    ax.tick_params(axis='x', rotation=45)
    apply_scientific_style(ax)
    for bar in bars:
        yval = bar.get_height()
        text_y_offset = 5000
        ha = 'center'
        va = 'bottom' if yval >= 0 else 'top'
        rotation_angle = 90
        ax.text(bar.get_x() + bar.get_width()/2, yval + (text_y_offset if yval >= 0 else -text_y_offset), f'€{yval:,.0f}',
                 ha=ha, va=va, fontsize=8, color='black', rotation=rotation_angle)
    fig.tight_layout() # Added tight_layout
    return fig

def generate_plot8(total_paths, median_final_wealth, percentile_5, percentile_90_wealth, end_date_str):
    fig, ax = plt.subplots(figsize=(12, 10))
    final_wealth_values = total_paths[:, -1]
    sorted_final_wealth = np.sort(final_wealth_values)
    cdf = np.arange(1, len(sorted_final_wealth) + 1) / len(sorted_final_wealth)

    ax.plot(sorted_final_wealth, cdf, color='darkgreen', linewidth=2)
    ax.set_xlabel('Final Total Wealth (€)')
    ax.set_ylabel('Cumulative Probability')
    end_date_for_title = datetime.strptime(end_date_str, '%Y-%m-%d').strftime('%b %Y')
    ax.set_title(f'Cumulative Distribution of Final Total Wealth ({end_date_for_title})')
    ax.grid(True, which='major', linestyle='-', linewidth='0.7', color='lightgray', alpha=0.8)
    ax.grid(True, which='minor', linestyle=':', linewidth='0.5', color='lightgray', alpha=0.5)
    ax.set_ylim(0, 1)

    ax.axvline(median_final_wealth, color='red', linestyle='--', label=f'Median: €{median_final_wealth:,.0f}')
    ax.axvline(percentile_5, color='orange', linestyle='--', label=f'5th Percentile: €{percentile_5:,.0f}')
    ax.axvline(percentile_90_wealth, color='blue', linestyle='--', label=f'90th Percentile: €{percentile_90_wealth:,.0f}')
    ax.legend(loc='upper left', fontsize=9, frameon=True, shadow=True, fancybox=True)

    apply_scientific_style(ax)
    fig.tight_layout() # Added tight_layout
    return fig




'''
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
        apply_scientific_style(ax3)
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
        apply_scientific_style(ax4)
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
        apply_scientific_style(ax5)
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
        apply_scientific_style(ax7)
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
        '''
    #Kein Plan