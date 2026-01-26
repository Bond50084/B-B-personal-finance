import io
import base64
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.gridspec as gridspec # Add this import at the top of main_app.py
from html.parser import HTMLParser
from simulator_app.simulation_v2_inflation import run_simulation
import traceback
import matplotlib
matplotlib.use('Agg') # Use the 'Agg' backend for non-interactive plotting
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
import json
import os
import tempfile # For creating temporary files/directories
import uuid 
import pickle # For serializing Python objects
from flask import Flask, render_template, request, redirect, url_for, session, send_file
from plot_generator import plot_to_base64, generate_plot1, generate_plot2, generate_plot3, generate_plot4, generate_plot5, generate_plot6, generate_plot7, generate_plot8
import plot_generator
from matplotlib.backends.backend_pdf import PdfPages

from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user, current_user
from flask_bcrypt import Bcrypt
from flask_admin import Admin
from flask_admin.contrib.sqla import ModelView
from flask_login import login_required, current_user




app = Flask(__name__)
app.secret_key = 'some_key' # In Produktion bitte ändern!

# --- 1. DATENBANK & SECURITY KONFIGURATION ---
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

# --- 2. LOGIN MANAGER KONFIGURATION ---
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login' # Wenn User nicht eingeloggt ist, hierhin schicken

# --- 3. DAS USER MODEL (Muss VOR dem Admin-Teil stehen!) ---
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    has_license = db.Column(db.Boolean, default=False) 

# --- NEUES MODELL: FÖRDERMITTEL ANFRAGEN ---
class FoerderRequest(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    # Falls der User eingeloggt ist, speichern wir die ID (optional)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    
    company_name = db.Column(db.String(150), nullable=False)
    industry = db.Column(db.String(100))
    description = db.Column(db.Text) # Hier beschreibt er das Vorhaben
    contact_email = db.Column(db.String(150))
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


# --- ADD THIS TO app.py (After the User class) ---

class Customer(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    first_name = db.Column(db.String(100), nullable=False)
    last_name = db.Column(db.String(100), nullable=False)
    # This column stores all simulation inputs (cash, rent, dates) as a text string (JSON)
    simulation_data = db.Column(db.Text, nullable=True) 
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Link to the User so we can do current_user.customers
    user = db.relationship('User', backref=db.backref('customers', lazy=True))


# User Loader für Flask-Login
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/my-account')
@login_required
def user_home():
    return render_template('user_home.html')


# --- NEW ROUTES FOR CLIENT MANAGEMENT ---

@app.route('/add-customer', methods=['GET', 'POST'])
@login_required
def add_customer():
    if request.method == 'POST':
        first_name = request.form.get('first_name')
        last_name = request.form.get('last_name')
        
        # Create new customer
        new_customer = Customer(first_name=first_name, last_name=last_name, user_id=current_user.id)
        
        # Optional: Set default simulation data so the form isn't empty
        default_data = {
            'initial_cash': 400000,
            'invest_frac': 0.8,
            'start_date': '2025-01-31',
            # ... you can add more defaults here if you like
        }
        new_customer.simulation_data = json.dumps(default_data)
        
        db.session.add(new_customer)
        db.session.commit()
        
        # Redirect directly to their simulation page
        return redirect(url_for('customer_simulation', customer_id=new_customer.id))
        
    return render_template('add_customer.html')

@app.route('/customer/<int:customer_id>', methods=['GET', 'POST'])
@login_required
def customer_simulation(customer_id):
    customer = Customer.query.get_or_404(customer_id)
    
    # Security check: Ensure this customer belongs to the logged-in user
    if customer.user_id != current_user.id:
        return "Unauthorized Access", 403

    # Load saved data (convert JSON string back to Python dictionary)
    saved_data = {}
    if customer.simulation_data:
        try:
            saved_data = json.loads(customer.simulation_data)
        except:
            saved_data = {}

    return render_template('simulator_parameters.html', customer=customer, saved_data=saved_data)

@app.route('/customer/<int:customer_id>/delete')
@login_required
def delete_customer(customer_id):
    customer = Customer.query.get_or_404(customer_id)
    if customer.user_id == current_user.id:
        db.session.delete(customer)
        db.session.commit()
    return redirect(url_for('user_home'))






# --- 4. ADMIN DASHBOARD (Muss NACH dem User Model stehen) ---
# Falls 'template_mode' Fehler wirft, entferne ", template_mode='bootstrap3'"
admin = Admin(app, name='Mein Dashboard')#, template_mode='bootstrap3')
admin.add_view(ModelView(User, db.session))

# --- 5. AUTH ROUTES (LOGIN / REGISTER / LOGOUT) ---

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        user = User.query.filter_by(email=email).first()
        
        # Prüfen ob User existiert UND Passwort stimmt
        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('simulator_index'))
        else:
            return "Falsches Passwort oder Email", 401
            
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        # Passwort hashen (verschlüsseln)
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        
        # Neuen User erstellen
        new_user = User(email=email, password=hashed_password, has_license=False)
        db.session.add(new_user)
        db.session.commit()
        
        return redirect(url_for('login'))
        
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('landing'))

# --- NEUE PRODUKTSEITEN ---

@app.route('/products/foerder-match', methods=['GET', 'POST'])
def foerder_match():
    success = False
    
    if request.method == 'POST':
        # Daten aus dem Formular holen
        company = request.form.get('company_name')
        industry = request.form.get('industry')
        desc = request.form.get('description')
        email = request.form.get('email')
        
        # In Datenbank speichern
        new_req = FoerderRequest(
            company_name=company,
            industry=industry,
            description=desc,
            contact_email=email,
            user_id=current_user.id if current_user.is_authenticated else None
        )
        db.session.add(new_req)
        db.session.commit()
        
        success = True # Damit wir im HTML "Danke!" anzeigen können

    return render_template('foerder_match.html', active_page='foerder_match', success=success)

@app.route('/presentation')
def presentation():
    # Make sure to pass the active_page if you want a tab highlighted, 
    # otherwise just leave it empty.
    return render_template('presentation.html')


@app.route('/products/esg-reader')
def esg_reader():
    # Du brauchst noch eine esg_reader.html
    return render_template('contact.html', active_page='products')

###Ende neuer Flask ROuten für Login und so 




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

def generate_pdf_report(
    output_buffer,
    simulation_summary_html,
    t, median_cash, median_port, p5_total, p80_total, median_total, threshold_cash, plot_params_text,
    cum_first_person, cum_second_person, cum_expense, cum_income, cum_net, text_plot_2_params,
    df,
    total_paths, median_final_wealth, percentile_5, percentile_90_wealth, end_date_str,
    prob_below_threshold, prob_zero_cash
):
    with PdfPages(output_buffer) as pdf:
        # Add a title page
        fig_title = plt.figure(figsize=(8.27, 11.69), dpi=300) # A4 size
        ax_title = fig_title.add_subplot(111)
        ax_title.text(0.5, 0.7, 'Financial Simulation Report',
                      horizontalalignment='center',
                      verticalalignment='center',
                      fontsize=24,
                      fontweight='bold',
                      transform=ax_title.transAxes)
        ax_title.text(0.5, 0.5, f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M")}',
                      horizontalalignment='center',
                      verticalalignment='center',
                      fontsize=14,
                      transform=ax_title.transAxes)
        ax_title.axis('off')
        pdf.savefig(fig_title, bbox_inches='tight')
        plt.close(fig_title)

        # Add the simulation summary (as plain text or a simplified HTML rendering)
        # For full HTML rendering with styles, you'd need a more advanced PDF library.
        # Here, we'll try to convert basic HTML to text for simplicity.
        summary_text = simulation_summary_html.replace('<br>', '\n').replace('<strong>', '').replace('</strong>', '')
        fig_summary = plt.figure(figsize=(8.27, 11.69), dpi=300) # A4 size
        ax_summary = fig_summary.add_subplot(111)
        ax_summary.text(0.05, 0.95, "Simulation Summary:",
                        horizontalalignment='left', verticalalignment='top',
                        fontsize=16, fontweight='bold', transform=ax_summary.transAxes)
        ax_summary.text(0.05, 0.90, summary_text,
                        horizontalalignment='left', verticalalignment='top',
                        fontsize=10, wrap=True, transform=ax_summary.transAxes)
        ax_summary.axis('off')
        pdf.savefig(fig_summary, bbox_inches='tight')
        plt.close(fig_summary)

        # Generate and add plots from the plot_generator module
        figures_to_pdf = {}

        # Plot 1: Key Financial Trajectories with 5–80%ile Shading
        figures_to_pdf['plot1'] = plot_generator.generate_plot1(t, median_cash, median_port, p5_total, p80_total, median_total, threshold_cash, plot_params_text)

        # Plot 2: Cumulative Cash Components Over Time
        figures_to_pdf['plot2'] = plot_generator.generate_plot2(t, cum_first_person, cum_second_person, cum_expense, cum_income, cum_net, threshold_cash, text_plot_2_params)

        # Plot 3: Monthly Cash Flow Components Over Time
        figures_to_pdf['plot3'] = plot_generator.generate_plot3(df)

        # Plot 4: Histogram of Final Total Wealth
        figures_to_pdf['plot4'] = plot_generator.generate_plot4(total_paths, median_final_wealth, percentile_5, end_date_str)

        # Plot 5: Probability of Cash Account Reaching Critical Levels Over Time
        figures_to_pdf['plot5'] = plot_generator.generate_plot5(t, prob_below_threshold, prob_zero_cash, threshold_cash)

        # Plot 6: Median Total Wealth Breakdown: Cash vs. Portfolio
        figures_to_pdf['plot6'] = plot_generator.generate_plot6(t, median_cash, median_port, median_total)

        # Plot 7: Annual Net Cash Flow
        figures_to_pdf['plot7'] = plot_generator.generate_plot7(df)

        # Plot 8: Cumulative Distribution of Final Total Wealth
        figures_to_pdf['plot8'] = plot_generator.generate_plot8(total_paths, median_final_wealth, percentile_5, percentile_90_wealth, end_date_str)

        # Add all generated figures to the PDF
        for fig_key in sorted(figures_to_pdf.keys()): # Ensure plots are added in order
            pdf.savefig(figures_to_pdf[fig_key], bbox_inches='tight')
            plt.close(figures_to_pdf[fig_key]) # Close the f



# Define route for the main website's index page
@app.route('/')
def landing():
    # This renders the new "Startup" introduction page
    return render_template('landing.html', active_page='home')

# --- The Old Main Page (Now "Products" or "Dashboard") ---
@app.route('/products')

def main_index():
    # This renders your existing product overview
    return render_template('main_index.html', active_page='products')

@app.route('/simulator')
@login_required  # <--- DIESER BEFEHL SCHÜTZT DIE SEITE!
def simulator_index():
    # Check: Hat der User auch bezahlt?
    if not current_user.has_license:
        return "Bitte kaufen Sie erst eine Lizenz!", 403
        
    return render_template('simulator_parameters.html', active_page='simulator')


@app.route('/contact')
def contact():
    return render_template('contact.html', active_page='contact')

@app.route('/simulator/documentation')
def simulator_documentation():
    # This would be a new page for your documentation
    return render_template('documentation.html', active_page='documentation') # You'll need to create documentation.html

@app.route('/simulator/save', methods=['POST'])
@login_required
def save_customer_data():
    customer_id = request.form.get('customer_id')
    
    if customer_id:
        customer = Customer.query.get_or_404(customer_id)
        
        # Security: Ensure the customer belongs to the current user
        if customer.user_id != current_user.id:
            return "Unauthorized Access", 403

        # Capture all form data and save it
        form_data = request.form.to_dict()
        customer.simulation_data = json.dumps(form_data)
        db.session.commit()
        
        # Redirect back to the edit page
        return redirect(url_for('customer_simulation', customer_id=customer.id))
    
    # If no customer ID (shouldn't happen if button is hidden), go back to index
    return redirect(url_for('simulator_index'))


@app.route('/simulator/run', methods=['POST'])
@login_required
def run():
    """
    Handles the simulation request, runs the simulation, generates plots,
    and displays results.
    """
    customer_id = request.form.get('customer_id')
    if customer_id:
        customer = Customer.query.get(customer_id)
        if customer and customer.user_id == current_user.id:
            # Dump the entire form data into the database as JSON
            # We convert the ImmutableMultiDict to a regular dict
            form_data = request.form.to_dict()
            customer.simulation_data = json.dumps(form_data)
            db.session.commit()

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
        #from flask import session
        simulation_data = {
            't': t.tolist(), # Convert numpy array to list
            'median_total': median_total.tolist(),
            'df_data': df.to_dict('records'), # Convert DataFrame to list of dicts
            'df_index_dates': df.index.strftime('%Y-%m-%d').tolist(),
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
            'percentile_5': percentile_5,
        }
         # Create a temporary file to store results
        temp_dir = tempfile.gettempdir() # Get the system's temporary directory
        unique_filename = f"simulation_results_{uuid.uuid4()}.json"
        temp_file_path = os.path.join(temp_dir, unique_filename)

        with open(temp_file_path, 'w') as f:
            json.dump(simulation_data, f)

        # Store the path to the temporary file in the session
        session['simulation_results_file'] = temp_file_path
        
    
        plots = {}

        # --- Plot 1: Key Financial Trajectories with 5–80%ile Shading ---
        #fig1, ax1 = plt.subplots(figsize=(10, 6))
        t_years = t / 12.0
        
        fig1 = generate_plot1(t, median_cash, median_port, p5_total, p80_total, median_total, threshold_cash, plot_params_text)
        plots['plot1'] = plot_to_base64(fig1)
        
        fig2 = generate_plot2(t, cum_first_person, cum_second_person, cum_expense, cum_income, cum_net, threshold_cash, text_plot_2)
        plots['plot2'] = plot_to_base64(fig2)
        
        fig3 = generate_plot3(df)
        plots['plot3'] = plot_to_base64(fig3)

        # Plot 4: Histogram of Final Total Wealth
        fig4 = generate_plot4(total_paths, median_final_wealth, percentile_5, end_date_str)
        plots['plot4'] = plot_to_base64(fig4)

        # Plot 5: Probability of Cash Account Reaching Critical Levels Over Time
        fig5 = generate_plot5(t, prob_below_threshold, prob_zero_cash, threshold_cash)
        plots['plot5'] = plot_to_base64(fig5)

        # Plot 6: Median Total Wealth Breakdown: Cash vs. Portfolio
        fig6 = generate_plot6(t, median_cash, median_port, median_total)
        plots['plot6'] = plot_to_base64(fig6)

        # Plot 7: Annual Net Cash Flow
        fig7 = generate_plot7(df)
        plots['plot7'] = plot_to_base64(fig7)

        # Plot 8: Cumulative Distribution of Final Total Wealth
        fig8 = generate_plot8(total_paths, median_final_wealth, percentile_5, percentile_90_wealth, end_date_str)
        plots['plot8'] = plot_to_base64(fig8)   
        
        
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
        
        #return render_template('simulator_parameters.html', error=f"An unexpected error occurred: {e}")
        return render_template('simulator_parameters.html', error=f"An unexpected error occurred: {e}<br><pre>{traceback.format_exc()}</pre>")


@app.route('/simulator/download_report', methods=['GET'])
def download_report():
    """
    Initiates the download of a PDF report based on the last simulation results
    stored temporarily on the server's file system.
    """
    temp_file_path = session.get('simulation_results_file')

    # If no file path in session or file does not exist, redirect with an error
    if not temp_file_path or not os.path.exists(temp_file_path):
        return redirect(url_for('simulator_index', error="No simulation results found. Please run a simulation first."))

    simulation_results = None
    pdf_output_buffer = io.BytesIO() # Create an in-memory buffer for the PDF

    try:
        # Load the simulation results from the temporary JSON file
        with open(temp_file_path, 'r') as f:
            simulation_results = json.load(f)

        # Reconstruct data from loaded JSON into NumPy arrays and Pandas DataFrames
        t = np.array(simulation_results['t'])
        median_total = np.array(simulation_results['median_total'])
        df = pd.DataFrame(simulation_results['df_data'])
        # IMPORTANT: Reconstruct the datetime index for the DataFrame
        df.index = pd.to_datetime(simulation_results['df_index_dates'])

        p5_total = np.array(simulation_results['p5_total'])
        p80_total = np.array(simulation_results['p80_total'])
        prob_below_threshold = np.array(simulation_results['prob_below_threshold'])
        prob_zero_cash = np.array(simulation_results['prob_zero_cash'])
        cum_income = np.array(simulation_results['cum_income'])
        cum_expense = np.array(simulation_results['cum_expense'])
        cum_net = np.array(simulation_results['cum_net'])
        cum_first_person = np.array(simulation_results['cum_first_person'])
        cum_second_person = np.array(simulation_results['cum_second_person'])
        median_cash = np.array(simulation_results['median_cash'])
        median_port = np.array(simulation_results['median_port'])
        total_paths = np.array(simulation_results['total_paths'])
        cum_large_payments = np.array(simulation_results['cum_large_payments'])

        # Retrieve direct values for summary and plot text
        median_final_wealth = simulation_results['median_final_wealth']
        median_final_wealth_after_taxes = simulation_results['median_final_wealth_after_taxes']
        median_final_wealth_today_s_power = simulation_results['median_final_wealth_today_s_power']
        median_final_wealth_today_s_power_before_taxes = simulation_results['median_final_wealth_today_s_power_before_taxes']
        prob_zero_cash_final = simulation_results['prob_zero_cash_final']
        percentile_90_wealth = simulation_results['percentile_90_wealth']
        simulation_summary_html = simulation_results['simulation_summary_html'] # Use the pre-generated HTML summary
        plot_params_text = simulation_results['plot_params_text']
        text_plot_2_params = simulation_results['text_plot_2_params']
        threshold_cash = simulation_results['threshold_cash']
        percentile_5 = simulation_results['percentile_5']
        start_date_str = simulation_results['start_date_str'] # Needed for date recreation if used in plots
        end_date_str = simulation_results['end_date_str'] # Needed for plot titles

        # Call the refactored PDF generation function
        generate_pdf_report(
            pdf_output_buffer,
            simulation_summary_html,
            t, median_cash, median_port, p5_total, p80_total, median_total, threshold_cash, plot_params_text,
            cum_first_person, cum_second_person, cum_expense, cum_income, cum_net, text_plot_2_params,
            df,
            total_paths, median_final_wealth, percentile_5, percentile_90_wealth, end_date_str,
            prob_below_threshold, prob_zero_cash
        )
        pdf_output_buffer.seek(0) # Rewind the buffer to the beginning for sending

        # Send the file to the user
        response = send_file(
            pdf_output_buffer,
            as_attachment=True,
            download_name="financial_simulation_report.pdf",
            mimetype='application/pdf'
        )
        return response

    except Exception as e:
        # Log the error for debugging
        print(f"Error generating PDF report: {e}")
        print(traceback.format_exc()) # Print the full traceback

        # Return a plain text error message to the user
        return f"Error generating your report. Please try running the simulation again. Details: {e}", 500
    finally:
        # Clean up: remove the temporary file and clear the session entry
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except OSError as e:
                print(f"Error deleting temporary file {temp_file_path}: {e}")
        if 'simulation_results_file' in session:
            session.pop('simulation_results_file', None)



if __name__ == '__main__':
    # Erst die Datenbank erstellen...
    with app.app_context():
        db.create_all()
        print("Datenbank wurde geprüft/erstellt.")

    # ...dann den Server starten (nur einmal!)
    app.run(debug=True)

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