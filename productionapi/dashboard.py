import pandas as pd
import numpy as np
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from datetime import datetime, timedelta
import os

# Custom CSS styling
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# Global variable to track data version
data_version = 0

def increment_data_version():
    """Call this function when data is updated to trigger dashboard refresh"""
    global data_version
    data_version += 1
    print(f"Data version incremented to: {data_version}")

def load_honey_production_data():
    """Load honey production data from CSV file"""
    try:
        # Check if file exists
        if not os.path.exists('honey_production_data.csv'):
            print("CSV file not found, creating sample data for demonstration")
            return create_sample_data()
        
        df = pd.read_csv('honey_production_data.csv')
        
        # If CSV is empty, return sample data
        if df.empty:
            print("CSV file is empty, creating sample data")
            return create_sample_data()
        
        # Convert numeric columns - ADD ERROR HANDLING
        numeric_cols = [
            'gross_weight_kg', 'drum_weight_kg', 'net_weight_kg', 'beshara_kg',
            'production_kg', 'num_production_hives', 'production_per_hive_kg',
            'num_hive_supers', 'efficiency_ratio', 'waste_percentage'
        ]
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # FILL NaN values with 0 for numeric columns
                df[col] = df[col].fillna(0)
        
        # Convert date columns
        if 'extraction_date' in df.columns:
            df['extraction_date'] = pd.to_datetime(df['extraction_date'], errors='coerce')
        
        # DON'T filter out rows - just handle NaN values
        # df = df.dropna(subset=['production_kg'])  # REMOVE THIS LINE
        
        # Clean location names
        if 'location' in df.columns:
            df['location'] = df['location'].str.strip()
            df['location'] = df['location'].replace('Unknown', 'Not Specified')
            df['location'] = df['location'].fillna('Not Specified')  # ADD THIS
        
        print(f"Successfully loaded {len(df)} honey production records")
        return df
    
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return create_sample_data()

# 2. Fix the data store callback - replace the entire callback around line 560

def create_sample_data():
    """Create sample data for demonstration"""
    base_date = datetime.now() - timedelta(days=365)
    sample_dates = [base_date + timedelta(days=np.random.randint(0, 365)) for _ in range(15)]
    
    sample_data = {
        'batch_number': ['API4144', 'API4145', 'API4146'] * 5,
        'report_year': [2024] * 15,
        'company_name': ['MANAHIL LLC'] * 15,
        'apiary_number': [f'{i} - Location{i%3}' for i in range(1, 16)],
        'location': ['Kalba', 'Dubai', 'Abu Dhabi'] * 5,
        'gross_weight_kg': np.random.uniform(200, 600, 15),
        'drum_weight_kg': np.random.uniform(10, 35, 15),
        'net_weight_kg': np.random.uniform(180, 570, 15),
        'beshara_kg': np.random.uniform(2, 4, 15),
        'production_kg': np.random.uniform(175, 565, 15),
        'num_production_hives': np.random.randint(20, 120, 15),
        'production_per_hive_kg': np.random.uniform(2, 32, 15),
        'num_hive_supers': np.random.randint(15, 80, 15),
        'efficiency_ratio': np.random.uniform(0.5, 15, 15),
        'waste_percentage': np.random.uniform(0.5, 2, 15),
        'extraction_date': sample_dates
    }
    return pd.DataFrame(sample_data)

# Create the Dash app
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, title="Honey Production Intelligence Dashboard")

# Define custom styles
colors = {
    'background': '#f8f9fa',
    'text': '#2c3e50',
    'primary': '#3498db',
    'success': '#27ae60',
    'warning': '#f39c12',
    'danger': '#e74c3c',
    'info': '#17a2b8'
}

card_style = {
    'backgroundColor': 'white',
    'padding': '20px',
    'borderRadius': '10px',
    'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)',
    'margin': '10px',
    'textAlign': 'center'
}

# Define the layout
app.layout = html.Div([
    # Manual refresh button and status
    html.Div([
        html.Button(
            '🔄 Refresh Dashboard', 
            id='refresh-button',
            style={
                'backgroundColor': '#008CBA',
                'color': 'white',
                'padding': '10px 20px',
                'border': 'none',
                'borderRadius': '4px',
                'cursor': 'pointer',
                'fontSize': '16px',
                'margin': '10px'
            }
        ),
        html.Span(id='last-update-time', style={'marginLeft': '20px', 'fontWeight': 'bold', 'color': 'green'})
    ], style={'textAlign': 'right', 'margin': '10px'}),
    
    # Data store component to hold the current dataframe
    dcc.Store(id='data-store'),
    
    html.Div([
        html.H1("🍯 Honey Production Intelligence Dashboard", 
                style={'textAlign': 'center', 'color': colors['text'], 'marginBottom': '30px'}),
        
        # Status indicator
        html.Div([
            html.Div(id='status-indicator', style={
                'textAlign': 'center',
                'padding': '15px',
                'margin': '10px',
                'borderRadius': '10px',
                'backgroundColor': '#e8f5e8',
                'border': '2px solid #4CAF50',
                'fontSize': '16px',
                'fontWeight': 'bold'
            })
        ]),
        
        # Summary Statistics Card
        html.Div([
            html.Div([
                html.H4("📊 Data Overview", style={'color': colors['text'], 'marginBottom': '15px'}),
                html.Div(id='data-overview-content')
            ], style={**card_style, 'backgroundColor': '#e8f4f8', 'textAlign': 'left'})
        ], style={'margin': '20px 0'}),
        
        # Filters Row - Enhanced with date filters
        html.Div([
            html.Div([
                html.Label("Select Batch:", style={'fontWeight': 'bold'}),
                dcc.Dropdown(
                    id='batch-filter',
                    value='all',
                    clearable=False
                )
            ], style={'width': '18%', 'display': 'inline-block', 'margin': '10px'}),
            
            html.Div([
                html.Label("Select Location:", style={'fontWeight': 'bold'}),
                dcc.Dropdown(
                    id='location-filter',
                    value='all',
                    clearable=False
                )
            ], style={'width': '18%', 'display': 'inline-block', 'margin': '10px'}),
            
            html.Div([
                html.Label("Select Year:", style={'fontWeight': 'bold'}),
                dcc.Dropdown(
                    id='year-filter',
                    value='all',
                    clearable=False
                )
            ], style={'width': '18%', 'display': 'inline-block', 'margin': '10px'}),
            
            html.Div([
                html.Label("Select Season:", style={'fontWeight': 'bold'}),
                dcc.Dropdown(
                    id='season-filter',
                    value='all',
                    clearable=False
                )
            ], style={'width': '18%', 'display': 'inline-block', 'margin': '10px'}),
            
            html.Div([
                html.Label("Date Range:", style={'fontWeight': 'bold'}),
                dcc.DatePickerRange(
                    id='date-range-picker',
                    display_format='YYYY-MM-DD'
                )
            ], style={'width': '20%', 'display': 'inline-block', 'margin': '10px'})
        ], style={'textAlign': 'center', 'backgroundColor': colors['background'], 'padding': '20px', 'borderRadius': '10px'}),
        
        # KPI Cards Row - Enhanced with date-related KPIs
        html.Div([
            html.Div([
                html.H3(id='total-production', style={'color': colors['success'], 'margin': '0 0 10px 0'}),
                html.P("Total Production (KG)", style={'margin': 0, 'fontWeight': 'bold'})
            ], style={**card_style, 'backgroundColor': '#d4edda'}),
            
            html.Div([
                html.H3(id='total-hives', style={'color': colors['primary'], 'margin': '0 0 10px 0'}),
                html.P("Total Production Hives", style={'margin': 0, 'fontWeight': 'bold'})
            ], style={**card_style, 'backgroundColor': '#d1ecf1'}),
            
            html.Div([
                html.H3(id='avg-efficiency', style={'color': colors['warning'], 'margin': '0 0 10px 0'}),
                html.P("Avg Production/Hive (KG)", style={'margin': 0, 'fontWeight': 'bold'})
            ], style={**card_style, 'backgroundColor': '#fff3cd'}),
            
            html.Div([
                html.H3(id='total-locations', style={'color': colors['danger'], 'margin': '0 0 10px 0'}),
                html.P("Active Locations", style={'margin': 0, 'fontWeight': 'bold'})
            ], style={**card_style, 'backgroundColor': '#f8d7da'}),
            
            html.Div([
                html.H3(id='recent-harvests', style={'color': colors['info'], 'margin': '0 0 10px 0'}),
                html.P("Recent Harvests (30d)", style={'margin': 0, 'fontWeight': 'bold'})
            ], style={**card_style, 'backgroundColor': '#d1ecf1'})
        ], style={'display': 'flex', 'justifyContent': 'space-between', 'margin': '20px 0'}),
        
        # Date Analysis Row - New section for harvest date analytics
        html.Div([
            html.Div([
                html.H3("Harvest Timeline", style={'textAlign': 'center', 'color': colors['text']}),
                dcc.Graph(id='harvest-timeline-chart')
            ], style={
                'width': '48%', 
                'display': 'inline-block', 
                'verticalAlign': 'top',
                'backgroundColor': 'white', 
                'borderRadius': '10px', 
                'padding': '10px',
                'margin': '1%',
                'boxSizing': 'border-box'
            }),
            
            html.Div([
                html.H3("Seasonal Production Analysis", style={'textAlign': 'center', 'color': colors['text']}),
                dcc.Graph(id='seasonal-production-chart')
            ], style={
                'width': '48%', 
                'display': 'inline-block', 
                'verticalAlign': 'top',
                'backgroundColor': 'white', 
                'borderRadius': '10px', 
                'padding': '10px',
                'margin': '1%',
                'boxSizing': 'border-box'
            })
        ], style={'textAlign': 'left', 'whiteSpace': 'nowrap'}),
        
        # Monthly Trends Row - New section
        html.Div([
            html.Div([
                html.H3("Monthly Production Trends", style={'textAlign': 'center', 'color': colors['text']}),
                dcc.Graph(id='monthly-trends-chart')
            ], style={
                'width': '48%', 
                'display': 'inline-block', 
                'verticalAlign': 'top',
                'backgroundColor': 'white', 
                'borderRadius': '10px', 
                'padding': '10px',
                'margin': '1%',
                'boxSizing': 'border-box'
            }),
            
            html.Div([
                html.H3("Production by Batch", style={'textAlign': 'center', 'color': colors['text']}),
                dcc.Graph(id='batch-production-chart')
            ], style={
                'width': '48%', 
                'display': 'inline-block', 
                'verticalAlign': 'top',
                'backgroundColor': 'white', 
                'borderRadius': '10px', 
                'padding': '10px',
                'margin': '1%',
                'boxSizing': 'border-box'
            })
        ], style={'textAlign': 'left', 'whiteSpace': 'nowrap'}),
        
        # Production Analysis Row - Original charts
        html.Div([
            html.Div([
                html.H3("Production by Location", style={'textAlign': 'center', 'color': colors['text']}),
                dcc.Graph(id='location-production-chart')
            ], style={
                'width': '48%', 
                'display': 'inline-block', 
                'verticalAlign': 'top',
                'backgroundColor': 'white', 
                'borderRadius': '10px', 
                'padding': '10px',
                'margin': '1%',
                'boxSizing': 'border-box'
            }),
            
            html.Div([
                html.H3("Waste Analysis", style={'textAlign': 'center', 'color': colors['text']}),
                dcc.Graph(id='waste-analysis')
            ], style={
                'width': '48%', 
                'display': 'inline-block', 
                'verticalAlign': 'top',
                'backgroundColor': 'white', 
                'borderRadius': '10px', 
                'padding': '10px',
                'margin': '1%',
                'boxSizing': 'border-box'
            })
        ], style={'textAlign': 'left', 'whiteSpace': 'nowrap'}),
        
        # Detailed Analysis Row - Original charts
        html.Div([
            html.Div([
                html.H3("Top Performing Apiaries", style={'textAlign': 'center', 'color': colors['text']}),
                dcc.Graph(id='top-apiaries')
            ], style={
                'width': '98%', 
                'display': 'inline-block', 
                'verticalAlign': 'top',
                'backgroundColor': 'white', 
                'borderRadius': '10px', 
                'padding': '10px',
                'margin': '1%',
                'boxSizing': 'border-box'
            })
        ], style={'textAlign': 'left', 'whiteSpace': 'nowrap'}),
        
        # Data Table Section - Enhanced with toggle button
        html.Div([
            html.Div([
                html.H3("Detailed Production Data", style={'display': 'inline-block', 'marginRight': '20px', 'color': colors['text']}),
                html.Button(
                    'Show/Hide Details', 
                    id='toggle-table-button',
                    style={
                        'backgroundColor': colors['primary'],
                        'color': 'white',
                        'padding': '10px 15px',
                        'border': 'none',
                        'borderRadius': '5px',
                        'cursor': 'pointer',
                        'fontSize': '16px'
                    }
                )
            ], style={'display': 'flex', 'alignItems': 'center', 'margin': '20px', 'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '10px'}),
            
            html.Div([
                dash_table.DataTable(
                    id='production-table',
                    columns=[
                        {'name': 'Batch', 'id': 'batch_number'},
                        {'name': 'Apiary', 'id': 'apiary_number'},
                        {'name': 'Location', 'id': 'location'},
                        {'name': 'Harvest Date', 'id': 'extraction_date_str'},
                        {'name': 'Season', 'id': 'extraction_season'},
                        {'name': 'Days Since Harvest', 'id': 'days_since_extraction'},
                        {'name': 'Production (KG)', 'id': 'production_kg_str'},
                        {'name': 'Hives', 'id': 'num_production_hives'},
                        {'name': 'KG/Hive', 'id': 'production_per_hive_kg_str'},
                        {'name': 'Efficiency %', 'id': 'efficiency_ratio_str'},
                        {'name': 'Waste %', 'id': 'waste_percentage_str'}
                    ],
                    style_table={'overflowX': 'auto'},
                    style_cell={'textAlign': 'center', 'padding': '10px'},
                    style_header={'backgroundColor': colors['primary'], 'color': 'white', 'fontWeight': 'bold'},
                    style_data_conditional=[
                        {
                            'if': {'filter_query': '{production_per_hive_kg} > 10'},
                            'backgroundColor': '#d4edda',
                            'color': 'black',
                        },
                        {
                            'if': {'filter_query': '{waste_percentage} > 5'},
                            'backgroundColor': '#f8d7da',
                            'color': 'black',
                        },
                        {
                            'if': {'filter_query': '{days_since_extraction} < 30'},
                            'backgroundColor': '#e8f4f8',
                            'color': 'black',
                        }
                    ],
                    page_size=15,
                    filter_action="native",
                    sort_action="native"
                )
            ], id='table-container', style={'display': 'none', 'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '10px', 'margin': '10px'})
        ])
    ], style={'backgroundColor': colors['background'], 'minHeight': '100vh', 'padding': '20px'})
])



# Callback to load data on page load or manual refresh
@app.callback(
    [Output('data-store', 'data'),
     Output('last-update-time', 'children'),
     Output('status-indicator', 'children')],
    [Input('refresh-button', 'n_clicks')],
    prevent_initial_call=False
)
def update_data_store(n_clicks):
    print(f"Loading honey production data - Button clicks: {n_clicks}")
    df = load_honey_production_data()
    
    # Add derived date columns for analysis
    if not df.empty:
        # ENSURE ALL REQUIRED COLUMNS EXIST
        required_cols = ['batch_number', 'location', 'report_year', 'production_kg', 'num_production_hives']
        for col in required_cols:
            if col not in df.columns:
                if col == 'production_kg':
                    df[col] = 0
                elif col == 'num_production_hives':
                    df[col] = 0
                elif col == 'report_year':
                    df[col] = 2024
                else:
                    df[col] = 'Unknown'
        
        if 'extraction_date' in df.columns:
            # Handle date processing with error checking
            df['extraction_date'] = pd.to_datetime(df['extraction_date'], errors='coerce')
            
            # Only process dates that are valid
            valid_dates = df['extraction_date'].notna()
            
            df.loc[valid_dates, 'extraction_month'] = df.loc[valid_dates, 'extraction_date'].dt.month
            df.loc[valid_dates, 'extraction_month_name'] = df.loc[valid_dates, 'extraction_date'].dt.strftime('%B')
            df.loc[valid_dates, 'extraction_season'] = df.loc[valid_dates, 'extraction_date'].dt.month.map({
                12: 'Winter', 1: 'Winter', 2: 'Winter',
                3: 'Spring', 4: 'Spring', 5: 'Spring',
                6: 'Summer', 7: 'Summer', 8: 'Summer',
                9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
            })
            df.loc[valid_dates, 'days_since_extraction'] = (datetime.now() - df.loc[valid_dates, 'extraction_date']).dt.days
            
            # Fill NaN values for date-derived columns
            df['extraction_month'] = df['extraction_month'].fillna(1)
            df['extraction_month_name'] = df['extraction_month_name'].fillna('January')
            df['extraction_season'] = df['extraction_season'].fillna('Unknown')
            df['days_since_extraction'] = df['days_since_extraction'].fillna(999)
        else:
            # Add default date columns if extraction_date doesn't exist
            df['extraction_month'] = 1
            df['extraction_month_name'] = 'January'
            df['extraction_season'] = 'Unknown'
            df['days_since_extraction'] = 999
    
    # Convert DataFrame to dict for storage
    data_dict = df.to_dict('records') if not df.empty else []
    
    # Update timestamp
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Status message
    if df.empty:
        status_msg = "🍯 No honey production data available. Upload production records to see analytics."
    else:
        status_msg = f"🍯 Dashboard loaded! Showing {len(df)} records from {df['batch_number'].nunique()} unique batches across {df['location'].nunique() if 'location' in df.columns else 0} locations."
    
    return data_dict, f"Last Updated: {current_time}", status_msg


# Callback to update filter options and date range when data changes
@app.callback(
    [Output('batch-filter', 'options'),
     Output('location-filter', 'options'),
     Output('year-filter', 'options'),
     Output('season-filter', 'options'),
     Output('date-range-picker', 'min_date_allowed'),
     Output('date-range-picker', 'max_date_allowed'),
     Output('date-range-picker', 'start_date'),
     Output('date-range-picker', 'end_date'),
     Output('data-overview-content', 'children')],
    [Input('data-store', 'data')]
)
def update_filter_options(data):
    if not data:
        empty_options = [{'label': 'No Data Available', 'value': 'all'}]
        overview_content = [
            html.P("Total Records: 0", style={'margin': '5px 0'}),
            html.P("Active Batches: 0", style={'margin': '5px 0'}),
            html.P("Locations: 0", style={'margin': '5px 0'}),
            html.P("Date Range: N/A", style={'margin': '5px 0'}),
            html.P(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", style={'margin': '5px 0', 'fontSize': '12px', 'color': '#666'})
        ]
        return (empty_options, empty_options, empty_options, empty_options, 
                None, None, None, None, overview_content)
    
    df = pd.DataFrame(data)
    
    if df.empty:
        empty_options = [{'label': 'No Data Available', 'value': 'all'}]
        overview_content = [
            html.P("Total Records: 0", style={'margin': '5px 0'}),
            html.P("Active Batches: 0", style={'margin': '5px 0'}),
            html.P("Locations: 0", style={'margin': '5px 0'}),
            html.P("Date Range: N/A", style={'margin': '5px 0'}),
            html.P(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", style={'margin': '5px 0', 'fontSize': '12px', 'color': '#666'})
        ]
        return (empty_options, empty_options, empty_options, empty_options, 
                None, None, None, None, overview_content)
    
    # Convert date columns back to datetime
    if 'extraction_date' in df.columns:
        df['extraction_date'] = pd.to_datetime(df['extraction_date'], errors='coerce')
    
    # Filter options
    batch_options = [{'label': 'All Batches', 'value': 'all'}] + [{'label': batch, 'value': batch} for batch in sorted(df['batch_number'].unique())]
    
    location_options = [{'label': 'All Locations', 'value': 'all'}]
    if 'location' in df.columns:
        location_options += [{'label': loc, 'value': loc} for loc in sorted(df['location'].unique())]
    
    year_options = [{'label': 'All Years', 'value': 'all'}]
    if 'report_year' in df.columns:
        year_options += [{'label': year, 'value': year} for year in sorted(df['report_year'].unique())]
    
    season_options = [{'label': 'All Seasons', 'value': 'all'}] + [{'label': season, 'value': season} for season in ['Spring', 'Summer', 'Autumn', 'Winter']]
    
    # Date range
    min_date = max_date = start_date = end_date = None
    if 'extraction_date' in df.columns:
        valid_dates = df['extraction_date'].dropna()
        if not valid_dates.empty:
            min_date = valid_dates.min()
            max_date = valid_dates.max()
            start_date = min_date
            end_date = max_date
    
    # Data overview
    overview_content = [
        html.P(f"Total Records: {len(df)}", style={'margin': '5px 0'}),
        html.P(f"Active Batches: {df['batch_number'].nunique()}", style={'margin': '5px 0'}),
        html.P(f"Locations: {df['location'].nunique() if 'location' in df.columns else 0}", style={'margin': '5px 0'}),
        html.P(f"Date Range: {min_date.strftime('%Y-%m-%d') if min_date else 'N/A'} to {max_date.strftime('%Y-%m-%d') if max_date else 'N/A'}", style={'margin': '5px 0'}),
        html.P(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", style={'margin': '5px 0', 'fontSize': '12px', 'color': '#666'})
    ]
    
    return (batch_options, location_options, year_options, season_options,
            min_date, max_date, start_date, end_date, overview_content)

# Enhanced callbacks with date functionality and data store
@app.callback(
    [
        Output('total-production', 'children'),
        Output('total-hives', 'children'),
        Output('avg-efficiency', 'children'),
        Output('total-locations', 'children'),
        Output('recent-harvests', 'children'),
        Output('harvest-timeline-chart', 'figure'),
        Output('seasonal-production-chart', 'figure'),
        Output('monthly-trends-chart', 'figure'),
        Output('batch-production-chart', 'figure'),
        Output('location-production-chart', 'figure'),
        Output('waste-analysis', 'figure'),
        Output('top-apiaries', 'figure'),
        Output('production-table', 'data')
    ],
    [
        Input('data-store', 'data'),
        Input('batch-filter', 'value'),
        Input('location-filter', 'value'),
        Input('year-filter', 'value'),
        Input('season-filter', 'value'),
        Input('date-range-picker', 'start_date'),
        Input('date-range-picker', 'end_date')
    ]
)
def update_dashboard(data, batch_filter, location_filter, year_filter, season_filter, start_date, end_date):
    # Handle empty data
    if not data:
        empty_fig = px.scatter(title="No data available - Click refresh or upload production data")
        return ("0 KG", "0", "0.00 KG", "0", "0", empty_fig, empty_fig, empty_fig, empty_fig,
                empty_fig, empty_fig, empty_fig, [])
    
    # Convert data back to DataFrame
    df = pd.DataFrame(data)
    
    if df.empty:
        empty_fig = px.scatter(title="No data available - Click refresh or upload production data")
        return ("0 KG", "0", "0.00 KG", "0", "0", empty_fig, empty_fig, empty_fig, empty_fig,
                empty_fig, empty_fig, empty_fig, [])
    
    # Debug information
    print(f"Dashboard update: Processing {len(df)} records")
    print(f"Available columns: {df.columns.tolist()}")
    if 'production_kg' in df.columns:
        print(f"Production data sample: {df['production_kg'].head().tolist()}")
        print(f"Production data sum: {df['production_kg'].sum()}")
    
    # Convert date columns back to datetime with error handling
    if 'extraction_date' in df.columns:
        df['extraction_date'] = pd.to_datetime(df['extraction_date'], errors='coerce')
    
    # Store original dataframe before filtering
    original_df = df.copy()
    filtered_df = df.copy()
    
    # Apply filters step by step with debugging
    print(f"Starting with {len(filtered_df)} records")
    
    if batch_filter and batch_filter != 'all':
        if 'batch_number' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['batch_number'] == batch_filter]
            print(f"After batch filter ({batch_filter}): {len(filtered_df)} records")
    
    if location_filter and location_filter != 'all':
        if 'location' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['location'] == location_filter]
            print(f"After location filter ({location_filter}): {len(filtered_df)} records")
    
    if year_filter and year_filter != 'all':
        if 'report_year' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['report_year'] == year_filter]
            print(f"After year filter ({year_filter}): {len(filtered_df)} records")
    
    if season_filter and season_filter != 'all':
        if 'extraction_season' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['extraction_season'] == season_filter]
            print(f"After season filter ({season_filter}): {len(filtered_df)} records")
    
    # Date range filtering
    if start_date and end_date and 'extraction_date' in filtered_df.columns:
        try:
            if isinstance(start_date, str):
                start_date = pd.to_datetime(start_date)
            if isinstance(end_date, str):
                end_date = pd.to_datetime(end_date)
            
            date_mask = (filtered_df['extraction_date'] >= start_date) & (filtered_df['extraction_date'] <= end_date)
            filtered_df = filtered_df[date_mask]
            print(f"After date filter ({start_date} to {end_date}): {len(filtered_df)} records")
        except Exception as e:
            print(f"Date filtering error: {e}")
    
    # If filtering results in empty dataframe, use original data with warning
    if filtered_df.empty:
        print("Warning: No data after filtering - using original dataset")
        filtered_df = original_df
        filter_warning = " (showing all data - no matches for current filters)"
    else:
        filter_warning = ""
    
    # Ensure required columns exist with default values
    required_columns = {
        'production_kg': 0,
        'num_production_hives': 0,
        'production_per_hive_kg': 0,
        'location': 'Unknown',
        'batch_number': 'Unknown',
        'apiary_number': 'Unknown',
        'waste_percentage': 0,
        'efficiency_ratio': 0
    }
    
    for col, default_val in required_columns.items():
        if col not in filtered_df.columns:
            filtered_df[col] = default_val
        else:
            # Fill NaN values
            if col in ['production_kg', 'num_production_hives', 'production_per_hive_kg', 'waste_percentage', 'efficiency_ratio']:
                filtered_df[col] = pd.to_numeric(filtered_df[col], errors='coerce').fillna(default_val)
            else:
                filtered_df[col] = filtered_df[col].fillna(default_val)
    
    # Calculate KPIs with error handling
    try:
        total_production_val = filtered_df['production_kg'].sum()
        total_production = f"{total_production_val:.1f} KG{filter_warning}"
        print(f"Total production calculated: {total_production_val}")
    except Exception as e:
        print(f"Error calculating total production: {e}")
        total_production = "0 KG"
    
    try:
        total_hives_val = int(filtered_df['num_production_hives'].sum())
        total_hives = f"{total_hives_val:,}"
    except Exception as e:
        print(f"Error calculating total hives: {e}")
        total_hives = "0"
    
    try:
        avg_efficiency_val = filtered_df['production_per_hive_kg'].mean()
        avg_efficiency = f"{avg_efficiency_val:.2f} KG" if not pd.isna(avg_efficiency_val) else "0.00 KG"
    except Exception as e:
        print(f"Error calculating avg efficiency: {e}")
        avg_efficiency = "0.00 KG"
    
    try:
        total_locations_val = filtered_df['location'].nunique()
        total_locations = f"{total_locations_val}"
    except Exception as e:
        print(f"Error calculating total locations: {e}")
        total_locations = "0"
    
    # Calculate recent harvests (last 30 days)
    recent_harvests = "0"
    try:
        if 'days_since_extraction' in filtered_df.columns:
            recent_count = len(filtered_df[filtered_df['days_since_extraction'] <= 30])
            recent_harvests = f"{recent_count}"
    except Exception as e:
        print(f"Error calculating recent harvests: {e}")
    
    # Create charts with proper error handling
    
    # 1. Harvest Timeline Chart
    try:
        if 'extraction_date' in filtered_df.columns and len(filtered_df) > 0:
            # Remove rows with null dates for this chart
            timeline_data = filtered_df.dropna(subset=['extraction_date'])
            if len(timeline_data) > 0:
                timeline_fig = px.scatter(
                    timeline_data,
                    x='extraction_date',
                    y='production_kg',
                    size='num_production_hives',
                    color='location',
                    hover_data=['batch_number', 'apiary_number'],
                    title="Production Timeline by Harvest Date"
                )
                timeline_fig.update_layout(xaxis_title="Harvest Date", yaxis_title="Production (KG)")
            else:
                timeline_fig = px.scatter(title="No harvest date data available for timeline")
        else:
            timeline_fig = px.scatter(title="No harvest date data available")
    except Exception as e:
        print(f"Error creating timeline chart: {e}")
        timeline_fig = px.scatter(title="Error creating harvest timeline chart")
    
    # 2. Seasonal Production Chart
    try:
        if 'extraction_season' in filtered_df.columns and len(filtered_df) > 0:
            seasonal_data = filtered_df.dropna(subset=['extraction_season'])
            if len(seasonal_data) > 0:
                seasonal_summary = seasonal_data.groupby('extraction_season')['production_kg'].sum().reset_index()
                if len(seasonal_summary) > 0:
                    seasonal_fig = px.bar(
                        seasonal_summary,
                        x='extraction_season',
                        y='production_kg',
                        title="Total Production by Season",
                        color='production_kg',
                        color_continuous_scale='Viridis'
                    )
                    seasonal_fig.update_layout(xaxis_title="Season", yaxis_title="Production (KG)")
                else:
                    seasonal_fig = px.bar(title="No seasonal data to display")
            else:
                seasonal_fig = px.bar(title="No valid seasonal data available")
        else:
            seasonal_fig = px.bar(title="No seasonal data available")
    except Exception as e:
        print(f"Error creating seasonal chart: {e}")
        seasonal_fig = px.bar(title="Error creating seasonal chart")
    
    # 3. Monthly Trends Chart
    try:
        if 'extraction_month_name' in filtered_df.columns and len(filtered_df) > 0:
            monthly_data = filtered_df.dropna(subset=['extraction_month_name'])
            if len(monthly_data) > 0:
                monthly_summary = monthly_data.groupby('extraction_month_name').agg({
                    'production_kg': 'sum',
                    'production_per_hive_kg': 'mean'
                }).reset_index()
                
                if len(monthly_summary) > 0:
                    monthly_fig = go.Figure()
                    monthly_fig.add_trace(go.Bar(
                        x=monthly_summary['extraction_month_name'],
                        y=monthly_summary['production_kg'],
                        name='Total Production (KG)',
                        yaxis='y'
                    ))
                    monthly_fig.add_trace(go.Scatter(
                        x=monthly_summary['extraction_month_name'],
                        y=monthly_summary['production_per_hive_kg'],
                        mode='lines+markers',
                        name='Avg KG/Hive',
                        yaxis='y2'
                    ))
                    
                    monthly_fig.update_layout(
                        title="Monthly Production Trends",
                        xaxis_title="Month",
                        yaxis=dict(title="Total Production (KG)", side="left"),
                        yaxis2=dict(title="Average KG/Hive", side="right", overlaying="y"),
                        legend=dict(x=0.01, y=0.99)
                    )
                else:
                    monthly_fig = px.bar(title="No monthly data to display")
            else:
                monthly_fig = px.bar(title="No valid monthly data available")
        else:
            monthly_fig = px.bar(title="No monthly data available")
    except Exception as e:
        print(f"Error creating monthly trends chart: {e}")
        monthly_fig = px.bar(title="Error creating monthly trends chart")
    
    # 4. Batch Production Chart
    try:
        if len(filtered_df) > 0:
            batch_summary = filtered_df.groupby('batch_number')['production_kg'].sum().reset_index()
            if len(batch_summary) > 0:
                batch_fig = px.bar(
                    batch_summary, 
                    x='batch_number', 
                    y='production_kg',
                    title="Total Production by Batch",
                    color='production_kg',
                    color_continuous_scale='Viridis'
                )
                batch_fig.update_layout(showlegend=False, xaxis_title="Batch Number", yaxis_title="Production (KG)")
            else:
                batch_fig = px.bar(title="No batch data to display")
        else:
            batch_fig = px.bar(title="No batch data available")
    except Exception as e:
        print(f"Error creating batch chart: {e}")
        batch_fig = px.bar(title="Error creating batch chart")
    
    # 5. Location Production Chart
    try:
        if len(filtered_df) > 0:
            location_summary = filtered_df.groupby('location')['production_kg'].sum().reset_index()
            if len(location_summary) > 0 and location_summary['production_kg'].sum() > 0:
                location_fig = px.pie(
                    location_summary,
                    values='production_kg',
                    names='location',
                    title="Production Distribution by Location"
                )
            else:
                location_fig = px.pie(title="No location data to display")
        else:
            location_fig = px.pie(title="No location data available")
    except Exception as e:
        print(f"Error creating location chart: {e}")
        location_fig = px.pie(title="Error creating location chart")
    
    # 6. Waste Analysis
    try:
        if len(filtered_df) > 0 and 'waste_percentage' in filtered_df.columns:
            waste_data = filtered_df[filtered_df['waste_percentage'].notna()]
            if len(waste_data) > 0:
                waste_fig = px.box(
                    waste_data,
                    x='location',
                    y='waste_percentage',
                    title="Waste Percentage Distribution by Location"
                )
                waste_fig.update_layout(xaxis_title="Location", yaxis_title="Waste Percentage (%)")
            else:
                waste_fig = px.box(title="No waste data to display")
        else:
            waste_fig = px.box(title="No waste data available")
    except Exception as e:
        print(f"Error creating waste analysis chart: {e}")
        waste_fig = px.box(title="Error creating waste analysis chart")
    
    # 7. Top Performing Apiaries
    try:
        if len(filtered_df) > 0 and 'production_per_hive_kg' in filtered_df.columns:
            apiary_data = filtered_df[filtered_df['production_per_hive_kg'].notna()]
            if len(apiary_data) > 0:
                top_apiaries = apiary_data.nlargest(10, 'production_per_hive_kg')[['apiary_number', 'production_per_hive_kg', 'location']]
                if len(top_apiaries) > 0:
                    top_fig = px.bar(
                        top_apiaries,
                        x='production_per_hive_kg',
                        y='apiary_number',
                        orientation='h',
                        color='location',
                        title="Top 10 Performing Apiaries (KG per Hive)"
                    )
                    top_fig.update_layout(xaxis_title="Production per Hive (KG)", yaxis_title="Apiary")
                else:
                    top_fig = px.bar(title="No apiary performance data to display")
            else:
                top_fig = px.bar(title="No valid apiary data available")
        else:
            top_fig = px.bar(title="No apiary data available")
    except Exception as e:
        print(f"Error creating top apiaries chart: {e}")
        top_fig = px.bar(title="Error creating top apiaries chart")
    
    # 8. Table Data - Create formatted columns for display
    table_data = []
    try:
        if len(filtered_df) > 0:
            table_df = filtered_df.copy()
            
            # Create formatted columns that the table expects
            if 'extraction_date' in table_df.columns:
                table_df['extraction_date_str'] = table_df['extraction_date'].dt.strftime('%Y-%m-%d')
                table_df['extraction_date_str'] = table_df['extraction_date_str'].fillna('N/A')
            else:
                table_df['extraction_date_str'] = 'N/A'
            
            # Format numeric columns
            table_df['production_kg_str'] = table_df['production_kg'].round(2).astype(str)
            table_df['production_per_hive_kg_str'] = table_df['production_per_hive_kg'].round(2).astype(str)
            table_df['efficiency_ratio_str'] = table_df['efficiency_ratio'].round(2).astype(str) + '%'
            table_df['waste_percentage_str'] = table_df['waste_percentage'].round(2).astype(str) + '%'
            
            # Ensure all required columns exist
            required_table_columns = ['batch_number', 'apiary_number', 'location', 'extraction_season', 'days_since_extraction', 'num_production_hives']
            for col in required_table_columns:
                if col not in table_df.columns:
                    if col == 'days_since_extraction' or col == 'num_production_hives':
                        table_df[col] = 0
                    else:
                        table_df[col] = 'N/A'
                else:
                    table_df[col] = table_df[col].fillna('N/A' if col not in ['days_since_extraction', 'num_production_hives'] else 0)
            
            table_data = table_df.to_dict('records')
            print(f"Table data prepared: {len(table_data)} records")
            
    except Exception as e:
        print(f"Error creating table data: {e}")
        table_data = []
    
    print("Dashboard update completed successfully")
    
    return (
        total_production, total_hives, avg_efficiency, total_locations, recent_harvests,
        timeline_fig, seasonal_fig, monthly_fig,
        batch_fig, location_fig, waste_fig,
        top_fig, table_data
    )
@app.callback(
    [Output('table-container', 'style'), Output('toggle-table-button', 'children')],
    [Input('toggle-table-button', 'n_clicks')],
    [State('table-container', 'style')]
)
def toggle_table_visibility(n_clicks, current_style):
    if n_clicks is None:
        return {'display': 'none', 'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '10px', 'margin': '10px'}, 'Show Details'
    
    if current_style.get('display') == 'none':
        return {'display': 'block', 'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '10px', 'margin': '10px'}, 'Hide Details'
    else:
        return {'display': 'none', 'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '10px', 'margin': '10px'}, 'Show Details'

# Run the app
if __name__ == '__main__':
    print("Starting Enhanced Honey Production Intelligence Dashboard...")
    print(f"Dashboard will be available at: http://localhost:8050")
    app.run(debug=True)