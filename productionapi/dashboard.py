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

data_version = 0

def increment_data_version():
    """Call this function when data is updated to trigger dashboard refresh"""
    global data_version
    data_version += 1
    print(f"Data version incremented to: {data_version}")

def load_honey_production_data():
    """Load honey production data from CSV file for dashboard visualization"""
    try:
        # Check if file exists
        if not os.path.exists('honey_production_data.csv'):
            print("CSV file not found, creating empty DataFrame")
            return pd.DataFrame(columns=[
                'batch_number', 'report_year', 'company_name', 'apiary_number',
                'location', 'gross_weight_kg', 'drum_weight_kg', 'net_weight_kg',
                'beshara_kg', 'production_kg', 'num_production_hives',
                'production_per_hive_kg', 'num_hive_supers', 'efficiency_ratio',
                'waste_percentage', 'extraction_date'
            ]), 0
        
        # Get file modification time
        mod_time = os.path.getmtime('honey_production_data.csv')
        
        # Load the CSV file
        df = pd.read_csv('honey_production_data.csv')
        
        # If CSV is empty, return empty DataFrame with proper columns
        if df.empty:
            print("CSV file is empty")
            return df, mod_time
        
        # Convert numeric columns
        numeric_cols = [
            'gross_weight_kg', 'drum_weight_kg', 'net_weight_kg', 'beshara_kg',
            'production_kg', 'num_production_hives', 'production_per_hive_kg',
            'num_hive_supers', 'efficiency_ratio', 'waste_percentage'
        ]
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Convert date columns
        if 'extraction_date' in df.columns:
            df['extraction_date'] = pd.to_datetime(df['extraction_date'], errors='coerce')
        
        # Filter out rows with no production data
        df = df.dropna(subset=['production_kg'])
        
        # Clean location names
        df['location'] = df['location'].str.strip()
        df['location'] = df['location'].replace('Unknown', 'Not Specified')
        
        # Add derived date columns for analysis
        if not df.empty and 'extraction_date' in df.columns:
            df['extraction_month'] = df['extraction_date'].dt.month
            df['extraction_month_name'] = df['extraction_date'].dt.strftime('%B')
            df['extraction_season'] = df['extraction_date'].dt.month.map({
                12: 'Winter', 1: 'Winter', 2: 'Winter',
                3: 'Spring', 4: 'Spring', 5: 'Spring',
                6: 'Summer', 7: 'Summer', 8: 'Summer',
                9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
            })
            df['days_since_extraction'] = (datetime.now() - df['extraction_date']).dt.days
        
        print(f"Successfully loaded {len(df)} honey production records")
        return df, mod_time
    
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        # Return empty DataFrame with expected columns as fallback
        return pd.DataFrame(columns=[
            'batch_number', 'report_year', 'company_name', 'apiary_number',
            'location', 'gross_weight_kg', 'drum_weight_kg', 'net_weight_kg',
            'beshara_kg', 'production_kg', 'num_production_hives',
            'production_per_hive_kg', 'num_hive_supers', 'efficiency_ratio',
            'waste_percentage', 'extraction_date'
        ]), 0
# Load the data

# Create the Dash app
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "Honey Production Intelligence Dashboard"

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
    html.Div([
        html.H1("🍯 Honey Production Intelligence Dashboard", 
                style={'textAlign': 'center', 'color': colors['text'], 'marginBottom': '30px'}),
        
        # Summary Statistics Card
        html.Div([
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
        dcc.Store(id='data-store'),
             html.Div([
             html.H4("📊 Data Overview", style={'color': colors['text'], 'marginBottom': '15px'}),
             html.P("Total Records: Loading...", style={'margin': '5px 0'}),
             html.P("Active Batches: Loading...", style={'margin': '5px 0'}),
             html.P("Locations: Loading...", style={'margin': '5px 0'}),
             html.P("Date Range: Loading...", style={'margin': '5px 0'}),
             html.P(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", style={'margin': '5px 0', 'fontSize': '12px', 'color': '#666'})
], style={**card_style, 'backgroundColor': '#e8f4f8', 'textAlign': 'left'})
        ], style={'margin': '20px 0'}),
        
        # Filters Row - Enhanced with date filters
        html.Div([
            html.Div([
                html.Label("Select Batch:", style={'fontWeight': 'bold'}),
                dcc.Dropdown(
                 id='batch-filter',
                 options=[{'label': 'All Batches', 'value': 'all'}],
                  value='all',
                  clearable=False
)
            ], style={'width': '18%', 'display': 'inline-block', 'margin': '10px'}),
            
            html.Div([
                html.Label("Select Location:", style={'fontWeight': 'bold'}),
                dcc.Dropdown(
                id='location-filter',
                options=[{'label': 'All Locations', 'value': 'all'}],
                value='all',
                clearable=False
)
            ], style={'width': '18%', 'display': 'inline-block', 'margin': '10px'}),
            
            html.Div([
                html.Label("Select Year:", style={'fontWeight': 'bold'}),
                dcc.Dropdown(
                id='year-filter',
                options=[{'label': 'All Years', 'value': 'all'}],
                value='all',
                clearable=False
)
            ], style={'width': '18%', 'display': 'inline-block', 'margin': '10px'}),
            
            html.Div([
                html.Label("Select Season:", style={'fontWeight': 'bold'}),
                dcc.Dropdown(
                id='season-filter',
                options=[{'label': 'All Seasons', 'value': 'all'}] + 
            [{'label': season, 'value': season} for season in ['Spring', 'Summer', 'Autumn', 'Winter']],
                 value='all',
                 clearable=False
)
            ], style={'width': '18%', 'display': 'inline-block', 'margin': '10px'}),
            
            html.Div([
                html.Label("Date Range:", style={'fontWeight': 'bold'}),
                dcc.DatePickerRange(
                id='date-range-picker',
                start_date=datetime.now() - timedelta(days=365),
                end_date=datetime.now(),
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
                html.H3("Harvest Freshness Analysis", style={'textAlign': 'center', 'color': colors['text']}),
                dcc.Graph(id='freshness-analysis-chart')
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
            }),
            
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
            })
        ], style={'textAlign': 'left', 'whiteSpace': 'nowrap'}),
        
        # Efficiency Analysis Row - Original charts
        html.Div([
            html.Div([
                html.H3("Hive Efficiency Analysis", style={'textAlign': 'center', 'color': colors['text']}),
                dcc.Graph(id='efficiency-scatter')
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
                html.H3("Production Distribution", style={'textAlign': 'center', 'color': colors['text']}),
                dcc.Graph(id='production-distribution')
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
        
        # Performance Matrix - Full width
        html.Div([
            html.H3("Production Performance Matrix", style={'textAlign': 'center', 'color': colors['text']}),
            dcc.Graph(id='performance-heatmap')
        ], style={'margin': '20px', 'backgroundColor': 'white', 'borderRadius': '10px', 'padding': '20px'}),
        
        # Data Table Section - Enhanced with date columns
        html.Div([
            html.Div([
                html.H3("Detailed Production Data", style={'display': 'inline-block', 'marginRight': '20px', 'color': colors['text']}),
                html.Button(
                    'Show Details', 
                    id='toggle-table-button',
                    style={
                        'backgroundColor': colors['primary'],
                        'color': 'white',
                        'padding': '10px 15px',
                        'border': 'none',
                        'borderRadius': '5px',
                        'cursor': 'pointer'
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
                        {'name': 'Harvest Date', 'id': 'extraction_date', 'type': 'datetime'},
                        {'name': 'Season', 'id': 'extraction_season'},
                        {'name': 'Days Since Harvest', 'id': 'days_since_extraction'},
                        {'name': 'Production (KG)', 'id': 'production_kg', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                        {'name': 'Hives', 'id': 'num_production_hives'},
                        {'name': 'KG/Hive', 'id': 'production_per_hive_kg', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                        {'name': 'Efficiency %', 'id': 'efficiency_ratio', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                        {'name': 'Waste %', 'id': 'waste_percentage', 'type': 'numeric', 'format': {'specifier': '.2f'}}
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
     Output('last-update-time', 'children')],
    [Input('refresh-button', 'n_clicks')],
    prevent_initial_call=False
)
def update_data_store(n_clicks):
    print(f"=== REFRESH BUTTON CLICKED: {n_clicks} ===")
    
    # Check if CSV file exists and get its info
    csv_path = 'honey_production_data.csv'
    if os.path.exists(csv_path):
        file_size = os.path.getsize(csv_path)
        mod_time = os.path.getmtime(csv_path)
        mod_time_readable = datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')
        print(f"CSV file found - Size: {file_size} bytes, Modified: {mod_time_readable}")
    else:
        print("CSV file not found!")
        return [], f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - FILE NOT FOUND"
    
    # Load fresh data
    df, file_mod_time = load_honey_production_data()
    print(f"Loaded DataFrame shape: {df.shape}")
    print(f"DataFrame columns: {list(df.columns)}")
    
    if not df.empty:
        print(f"Sample data (first 2 rows):")
        print(df.head(2).to_string())
        print(f"Unique batch numbers: {sorted(df['batch_number'].unique())}")
    
    # Convert DataFrame to dict for storage
    data_dict = df.to_dict('records') if not df.empty else []
    print(f"Converted to {len(data_dict)} records for storage")
    
    # Update timestamp
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    return data_dict, f"Last Updated: {current_time} - Records: {len(data_dict)}"
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
        Output('freshness-analysis-chart', 'figure'),
        Output('batch-production-chart', 'figure'),
        Output('location-production-chart', 'figure'),
        Output('efficiency-scatter', 'figure'),
        Output('waste-analysis', 'figure'),
        Output('performance-heatmap', 'figure'),
        Output('top-apiaries', 'figure'),
        Output('production-distribution', 'figure'),
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
    # Convert data back to DataFrame
    df = pd.DataFrame(data) if data else pd.DataFrame()
    
    # Handle empty data case
    if df.empty:
        empty_fig = px.scatter(title="No data available - Click refresh or upload data")
        return ("0 KG", "0", "0.00 KG", "0", "0", empty_fig, empty_fig, empty_fig, empty_fig,
                empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, [])
    
    # Convert date column if it exists
    if 'extraction_date' in df.columns:
        df['extraction_date'] = pd.to_datetime(df['extraction_date'], errors='coerce')
        # Recalculate derived date columns
        df['extraction_month'] = df['extraction_date'].dt.month
        df['extraction_month_name'] = df['extraction_date'].dt.strftime('%B')
        df['extraction_season'] = df['extraction_date'].dt.month.map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
        })
        df['days_since_extraction'] = (datetime.now() - df['extraction_date']).dt.days
    
    # Filter data
    filtered_df = df.copy()
    
    if batch_filter and batch_filter != 'all':
        filtered_df = filtered_df[filtered_df['batch_number'] == batch_filter]
    if location_filter and location_filter != 'all':
        filtered_df = filtered_df[filtered_df['location'] == location_filter]
    if year_filter and year_filter != 'all':
        filtered_df = filtered_df[filtered_df['report_year'] == year_filter]
    if season_filter and season_filter != 'all' and 'extraction_season' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['extraction_season'] == season_filter]
    
    # Date range filtering
    if start_date and end_date and 'extraction_date' in filtered_df.columns:
        filtered_df = filtered_df[
            (filtered_df['extraction_date'] >= start_date) & 
            (filtered_df['extraction_date'] <= end_date)
        ]
    
    # Calculate KPIs
    total_production = f"{filtered_df['production_kg'].sum():.1f} KG"
    total_hives = f"{filtered_df['num_production_hives'].sum():,}"
    avg_efficiency = f"{filtered_df['production_per_hive_kg'].mean():.2f} KG"
    total_locations = f"{filtered_df['location'].nunique()}"
    
    # Calculate recent harvests (last 30 days)
    recent_harvests = "0"
    if 'days_since_extraction' in filtered_df.columns:
        recent_count = len(filtered_df[filtered_df['days_since_extraction'] <= 30])
        recent_harvests = f"{recent_count}"
    
    # 1. Harvest Timeline Chart
    if 'extraction_date' in filtered_df.columns and not filtered_df.empty:
        timeline_fig = px.scatter(
            filtered_df,
            x='extraction_date',
            y='production_kg',
            size='num_production_hives',
            color='location',
            hover_data=['batch_number', 'apiary_number'],
            title="Production Timeline by Harvest Date"
        )
        timeline_fig.update_layout(xaxis_title="Harvest Date", yaxis_title="Production (KG)")
    else:
        timeline_fig = px.scatter(title="No harvest date data available")
    
    # 2. Seasonal Production Chart
    if 'extraction_season' in filtered_df.columns and not filtered_df.empty:
        seasonal_summary = filtered_df.groupby('extraction_season')['production_kg'].sum().reset_index()
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
        seasonal_fig = px.scatter(title="No seasonal data available")
    
    # 3. Monthly Trends Chart
    if 'extraction_month_name' in filtered_df.columns and not filtered_df.empty:
        monthly_summary = filtered_df.groupby('extraction_month_name').agg({
            'production_kg': 'sum',
            'production_per_hive_kg': 'mean'
        }).reset_index()
        
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
        monthly_fig = px.scatter(title="No monthly data available")
    
    # 4. Freshness Analysis Chart
    if 'days_since_extraction' in filtered_df.columns and not filtered_df.empty:
        # Create freshness categories
        filtered_df['freshness_category'] = pd.cut(
            filtered_df['days_since_extraction'],
            bins=[0, 7, 30, 90, 365, float('inf')],
            labels=['Very Fresh (0-7d)', 'Fresh (8-30d)', 'Good (31-90d)', 'Aging (91-365d)', 'Old (>365d)']
        )
        
        freshness_summary = filtered_df.groupby('freshness_category')['production_kg'].sum().reset_index()
        freshness_fig = px.pie(
            freshness_summary,
            values='production_kg',
            names='freshness_category',
            title="Production Distribution by Harvest Freshness"
        )
    else:
        freshness_fig = px.scatter(title="No freshness data available")
    
    # Original charts (5-11) with same logic as before
    # 5. Batch Production Chart
    if not filtered_df.empty:
        batch_summary = filtered_df.groupby('batch_number')['production_kg'].sum().reset_index()
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
        batch_fig = px.scatter(title="No batch data available")
    
    # 6. Location Production Chart
    if not filtered_df.empty:
        location_summary = filtered_df.groupby('location')['production_kg'].sum().reset_index()
        location_fig = px.pie(
            location_summary,
            values='production_kg',
            names='location',
            title="Production Distribution by Location"
        )
    else:
        location_fig = px.scatter(title="No location data available")
    
    # 7. Efficiency Scatter Plot
    if not filtered_df.empty:
        efficiency_fig = px.scatter(
            filtered_df,
            x='num_production_hives',
            y='production_per_hive_kg',
            size='production_kg',
            color='location',
            hover_data=['batch_number', 'apiary_number'],
            title="Hive Efficiency: Production per Hive vs Number of Hives"
        )
        efficiency_fig.update_layout(xaxis_title="Number of Production Hives", yaxis_title="Production per Hive (KG)")
    else:
        efficiency_fig = px.scatter(title="No efficiency data available")
    
    # 8. Waste Analysis
    if not filtered_df.empty:
        waste_fig = px.box(
            filtered_df,
            x='location',
            y='waste_percentage',
            title="Waste Percentage Distribution by Location"
        )
        waste_fig.update_layout(xaxis_title="Location", yaxis_title="Waste Percentage (%)")
    else:
        waste_fig = px.scatter(title="No waste data available")
    
    # 9. Performance Heatmap
    if len(filtered_df) > 1 and filtered_df['location'].nunique() > 1:
        heatmap_data = filtered_df.groupby(['location', 'batch_number'])['production_per_hive_kg'].mean().reset_index()
        heatmap_pivot = heatmap_data.pivot(index='location', columns='batch_number', values='production_per_hive_kg')
        
        heatmap_fig = px.imshow(
            heatmap_pivot.values,
            x=heatmap_pivot.columns,
            y=heatmap_pivot.index,
            aspect="auto",
            title="Average Production per Hive: Location vs Batch",
            color_continuous_scale='RdYlGn'
        )
        heatmap_fig.update_layout(xaxis_title="Batch Number", yaxis_title="Location")
    else:
        heatmap_fig = px.scatter(title="Insufficient data for heatmap")
    
    # 10. Top Performing Apiaries
    if not filtered_df.empty:
        top_apiaries = filtered_df.nlargest(10, 'production_per_hive_kg')[['apiary_number', 'production_per_hive_kg', 'location']]
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
        top_fig = px.scatter(title="No apiary data available")
    
    # 11. Production Distribution
    if not filtered_df.empty:
        dist_fig = px.histogram(
            filtered_df,
            x='production_per_hive_kg',
            nbins=20,
            title="Distribution of Production per Hive",
            marginal="box"
        )
        dist_fig.update_layout(xaxis_title="Production per Hive (KG)", yaxis_title="Frequency")
    else:
        dist_fig = px.scatter(title="No distribution data available")
    
    # 12. Table Data
    table_data = filtered_df.to_dict('records') if not filtered_df.empty else []
    
    return (
        total_production, total_hives, avg_efficiency, total_locations, recent_harvests,
        timeline_fig, seasonal_fig, monthly_fig, freshness_fig,
        batch_fig, location_fig, efficiency_fig, waste_fig, heatmap_fig,
        top_fig, dist_fig, table_data
    )
@app.callback(
    [Output('batch-filter', 'options'),
     Output('location-filter', 'options'),
     Output('year-filter', 'options')],
    [Input('data-store', 'data')]
)
def update_filter_options(data):
    if not data:
        return [], [], []
    
    df = pd.DataFrame(data)
    if df.empty:
        return [], [], []
    
    batch_options = [{'label': 'All Batches', 'value': 'all'}] + \
                   [{'label': batch, 'value': batch} for batch in sorted(df['batch_number'].unique())]
    
    location_options = [{'label': 'All Locations', 'value': 'all'}] + \
                      [{'label': loc, 'value': loc} for loc in sorted(df['location'].unique())]
    
    year_options = [{'label': 'All Years', 'value': 'all'}] + \
                  [{'label': year, 'value': year} for year in sorted(df['report_year'].unique())]
    
    return batch_options, location_options, year_options
# Toggle table visibility callback
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
    app.run(debug=True)