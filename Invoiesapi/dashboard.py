import pandas as pd
import os
from datetime import datetime, timedelta
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Global variable to track data version
# Create the Dash app

data_version = 0
def increment_data_version():
    """Call this function when data is updated to trigger dashboard refresh"""
    global data_version
    data_version += 1
    print(f"Data version incremented to: {data_version}")

def load_invoice_data():
    """Load invoice data from CSV file for dashboard visualization"""
    try:
        # Check if file exists
        if not os.path.exists('invoice_data.csv'):
            print("CSV file not found, creating empty DataFrame")
            return pd.DataFrame(columns=[
                'invoice_id', 'invoice_date', 'customer_name', 'customer_id',
                'customer_location', 'customer_type', 'customer_trn', 'payment_status',
                'due_date', 'product', 'qty', 'unit_price', 'total', 'amount_excl_vat',
                'vat', 'profit', 'profit_margin', 'cost_price', 'days_to_payment'
            ])
        
        # Load the CSV file
        df = pd.read_csv('invoice_data.csv')
        
        # If CSV is empty, return empty DataFrame with proper columns
        if df.empty:
            print("CSV file is empty")
            return df
        
        # Convert date columns to datetime
        date_columns = ['invoice_date', 'due_date']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Ensure numeric columns are properly typed
        numeric_cols = ['qty', 'unit_price', 'total', 'amount_excl_vat', 'vat', 
                         'profit', 'profit_margin', 'cost_price', 'days_to_payment']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Calculate payment date based on invoice_date and days_to_payment if available
        if 'days_to_payment' in df.columns and df['days_to_payment'].notna().any():
            df['payment_date'] = df.apply(
                lambda x: x['invoice_date'] + timedelta(days=x['days_to_payment']) 
                if pd.notna(x['days_to_payment']) else None, 
                axis=1
            )
        
        print(f"Successfully loaded {len(df)} invoice records")
        return df
    
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        # Return empty DataFrame with expected columns as fallback
        return pd.DataFrame(columns=[
            'invoice_id', 'invoice_date', 'customer_name', 'customer_id',
            'customer_location', 'customer_type', 'customer_trn', 'payment_status',
            'due_date', 'product', 'qty', 'unit_price', 'total', 'amount_excl_vat',
            'vat', 'profit', 'profit_margin', 'cost_price', 'days_to_payment'
        ])


app = dash.Dash(__name__, 
                title="🍯 Honey Analytics Dashboard",
                requests_pathname_prefix='/dash_app/')
# Enhanced styling with honey theme
honey_colors = {
    'primary': '#D4A574',      # Golden honey
    'secondary': '#F4E4BC',    # Light honey cream
    'accent': '#8B4513',       # Dark honey/amber
    'success': '#228B22',      # Forest green
    'warning': '#FF8C00',      # Dark orange
    'background': '#FFF8DC',   # Cornsilk
    'card_bg': '#FFFEF7',      # Very light cream
    'text_dark': '#2F2F2F',    # Dark gray
    'text_light': '#666666'    # Medium gray
}

# Custom CSS styles
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #FFF8DC 0%, #F4E4BC 100%);
                margin: 0;
                padding: 0;
                min-height: 100vh;
            }
            .card-hover {
                transition: all 0.3s ease;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }
            .card-hover:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 16px rgba(0,0,0,0.15);
            }
            .gradient-text {
                background: linear-gradient(45deg, #D4A574, #8B4513);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Define the enhanced layout
app.layout = html.Div([
    # Header with refresh button
    html.Div([
        html.Div([
            html.H1([
                "🍯 ",
                html.Span("Honey Business Analytics", className="gradient-text")
            ], style={
                'textAlign': 'center',
                'color': honey_colors['accent'],
                'marginBottom': '10px',
                'fontSize': '2.5rem',
                'fontWeight': '700'
            }),
            html.P("Sweet insights for your honey business", style={
                'textAlign': 'center',
                'color': honey_colors['text_light'],
                'fontSize': '1.1rem',
                'marginTop': '0'
            })
        ], style={'flex': '1'}),
        
        html.Div([
            html.Button([
                "🔄 ",
                html.Span("Refresh Data")
            ], id='refresh-button', style={
                'background': f'linear-gradient(135deg, {honey_colors["primary"]} 0%, {honey_colors["accent"]} 100%)',
                'color': 'white',
                'padding': '12px 24px',
                'border': 'none',
                'borderRadius': '25px',
                'cursor': 'pointer',
                'fontSize': '16px',
                'fontWeight': '600',
                'boxShadow': '0 4px 15px rgba(212, 165, 116, 0.3)',
                'transition': 'all 0.3s ease',
                'display': 'flex',
                'alignItems': 'center',
                'gap': '8px'
            }),
            html.Div(id='last-update-time', style={
                'marginTop': '8px',
                'fontSize': '0.9rem',
                'color': honey_colors['success'],
                'fontWeight': '500'
            })
        ], style={'textAlign': 'right'})
    ], style={
        'display': 'flex',
        'justifyContent': 'space-between',
        'alignItems': 'center',
        'padding': '20px 40px',
        'background': f'linear-gradient(135deg, {honey_colors["card_bg"]} 0%, {honey_colors["secondary"]} 100%)',
        'borderRadius': '0 0 20px 20px',
        'marginBottom': '30px',
        'boxShadow': '0 4px 20px rgba(0,0,0,0.1)'
    }),
    
    # Data store component to hold the current dataframe
    dcc.Store(id='data-store'),
    
    # Status indicator
    html.Div([
        html.Div(id='status-indicator', style={
            'textAlign': 'center',
            'padding': '15px',
            'margin': '0 40px 30px 40px',
            'borderRadius': '15px',
            'background': f'linear-gradient(135deg, {honey_colors["success"]}15 0%, {honey_colors["success"]}25 100%)',
            'border': f'2px solid {honey_colors["success"]}',
            'fontSize': '1.1rem',
            'fontWeight': '500',
            'color': honey_colors['text_dark']
        })
    ]),
    
    # Date Range Filter
    html.Div([
        html.Div([
            html.H3("📅 Filter by Date Range", style={
                'color': honey_colors['accent'],
                'marginBottom': '15px',
                'fontSize': '1.3rem',
                'fontWeight': '600'
            }),
            dcc.DatePickerRange(
                id='date-range',
                display_format='YYYY-MM-DD',
                style={'marginTop': '10px'}
            ),
        ], style={
            'background': honey_colors['card_bg'],
            'padding': '20px',
            'borderRadius': '15px',
            'boxShadow': '0 4px 15px rgba(0,0,0,0.1)',
            'border': f'1px solid {honey_colors["secondary"]}'
        })
    ], style={'margin': '0 40px 30px 40px'}),
    
    # KPI Cards
    html.Div([
        html.Div([
            html.Div("💰", style={'fontSize': '2.5rem', 'marginBottom': '10px'}),
            html.H3("Total Revenue", style={'color': honey_colors['text_dark'], 'margin': '0', 'fontSize': '1.1rem'}),
            html.H2(id='total-revenue', style={'color': honey_colors['accent'], 'margin': '10px 0 0 0', 'fontSize': '1.8rem', 'fontWeight': '700'})
        ], className='card-hover', style={
            'background': f'linear-gradient(135deg, {honey_colors["card_bg"]} 0%, {honey_colors["secondary"]} 100%)',
            'border': f'2px solid {honey_colors["primary"]}',
            'padding': '25px',
            'borderRadius': '20px',
            'textAlign': 'center',
            'flex': '1',
            'margin': '0 10px'
        }),
        
        html.Div([
            html.Div("📊", style={'fontSize': '2.5rem', 'marginBottom': '10px'}),
            html.H3("Invoices Count", style={'color': honey_colors['text_dark'], 'margin': '0', 'fontSize': '1.1rem'}),
            html.H2(id='invoice-count', style={'color': honey_colors['accent'], 'margin': '10px 0 0 0', 'fontSize': '1.8rem', 'fontWeight': '700'})
        ], className='card-hover', style={
            'background': f'linear-gradient(135deg, {honey_colors["card_bg"]} 0%, {honey_colors["secondary"]} 100%)',
            'border': f'2px solid {honey_colors["warning"]}',
            'padding': '25px',
            'borderRadius': '20px',
            'textAlign': 'center',
            'flex': '1',
            'margin': '0 10px'
        }),
        
        html.Div([
            html.Div("📈", style={'fontSize': '2.5rem', 'marginBottom': '10px'}),
            html.H3("Average Invoice", style={'color': honey_colors['text_dark'], 'margin': '0', 'fontSize': '1.1rem'}),
            html.H2(id='avg-invoice', style={'color': honey_colors['accent'], 'margin': '10px 0 0 0', 'fontSize': '1.8rem', 'fontWeight': '700'})
        ], className='card-hover', style={
            'background': f'linear-gradient(135deg, {honey_colors["card_bg"]} 0%, {honey_colors["secondary"]} 100%)',
            'border': f'2px solid {honey_colors["success"]}',
            'padding': '25px',
            'borderRadius': '20px',
            'textAlign': 'center',
            'flex': '1',
            'margin': '0 10px'
        }),
        
        html.Div([
            html.Div("✅", style={'fontSize': '2.5rem', 'marginBottom': '10px'}),
            html.H3("Payment Rate", style={'color': honey_colors['text_dark'], 'margin': '0', 'fontSize': '1.1rem'}),
            html.H2(id='payment-rate', style={'color': honey_colors['accent'], 'margin': '10px 0 0 0', 'fontSize': '1.8rem', 'fontWeight': '700'})
        ], className='card-hover', style={
            'background': f'linear-gradient(135deg, {honey_colors["card_bg"]} 0%, {honey_colors["secondary"]} 100%)',
            'border': f'2px solid {honey_colors["accent"]}',
            'padding': '25px',
            'borderRadius': '20px',
            'textAlign': 'center',
            'flex': '1',
            'margin': '0 10px'
        })
    ], style={'display': 'flex', 'margin': '0 40px 40px 40px'}),
    
    # Revenue Trends
    html.Div([
        html.Div([
            html.H2("📈 Revenue Trends", style={
                'color': honey_colors['accent'],
                'marginBottom': '20px',
                'fontSize': '1.5rem',
                'fontWeight': '600'
            }),
            dcc.Graph(id='revenue-trend-graph')
        ], style={
            'background': honey_colors['card_bg'],
            'padding': '25px',
            'borderRadius': '20px',
            'boxShadow': '0 6px 20px rgba(0,0,0,0.1)',
            'border': f'1px solid {honey_colors["secondary"]}'
        })
    ], style={'margin': '0 40px 30px 40px'}),
    
    # Product Performance
    html.Div([
        html.Div([
            html.Div([
                html.H2("🍯 Product Distribution", style={
                    'color': honey_colors['accent'],
                    'marginBottom': '20px',
                    'fontSize': '1.4rem',
                    'fontWeight': '600'
                }),
                dcc.Graph(id='product-distribution')
            ], style={
                'background': honey_colors['card_bg'],
                'padding': '25px',
                'borderRadius': '20px',
                'boxShadow': '0 6px 20px rgba(0,0,0,0.1)',
                'border': f'1px solid {honey_colors["secondary"]}',
                'width': '48%'
            }),
            
            html.Div([
                html.H2("🏆 Top Products by Revenue", style={
                    'color': honey_colors['accent'],
                    'marginBottom': '20px',
                    'fontSize': '1.4rem',
                    'fontWeight': '600'
                }),
                dcc.Graph(id='product-revenue')
            ], style={
                'background': honey_colors['card_bg'],
                'padding': '25px',
                'borderRadius': '20px',
                'boxShadow': '0 6px 20px rgba(0,0,0,0.1)',
                'border': f'1px solid {honey_colors["secondary"]}',
                'width': '48%'
            })
        ], style={'display': 'flex', 'justifyContent': 'space-between'})
    ], style={'margin': '0 40px 30px 40px'}),
    
    # Customer Insights
    html.Div([
        html.H2("👥 Customer Insights", style={
            'color': honey_colors['accent'],
            'marginBottom': '25px',
            'fontSize': '1.6rem',
            'fontWeight': '600',
            'textAlign': 'center'
        }),
        html.Div([
            html.Div([
                html.H3("🗺️ Revenue by Location", style={
                    'color': honey_colors['text_dark'],
                    'marginBottom': '15px',
                    'fontSize': '1.2rem',
                    'fontWeight': '500'
                }),
                dcc.Graph(id='location-revenue')
            ], style={
                'background': honey_colors['card_bg'],
                'padding': '20px',
                'borderRadius': '15px',
                'boxShadow': '0 4px 15px rgba(0,0,0,0.08)',
                'border': f'1px solid {honey_colors["secondary"]}',
                'width': '48%'
            }),
            
            html.Div([
                html.H3("🏢 Revenue by Customer Type", style={
                    'color': honey_colors['text_dark'],
                    'marginBottom': '15px',
                    'fontSize': '1.2rem',
                    'fontWeight': '500'
                }),
                dcc.Graph(id='customer-type-revenue')
            ], style={
                'background': honey_colors['card_bg'],
                'padding': '20px',
                'borderRadius': '15px',
                'boxShadow': '0 4px 15px rgba(0,0,0,0.08)',
                'border': f'1px solid {honey_colors["secondary"]}',
                'width': '48%'
            })
        ], style={'display': 'flex', 'justifyContent': 'space-between'})
    ], style={'margin': '0 40px 30px 40px'}),
    
    # Financial Analysis
    html.Div([
        html.Div([
            html.H2("💹 Profit Analysis", style={
                'color': honey_colors['accent'],
                'marginBottom': '20px',
                'fontSize': '1.5rem',
                'fontWeight': '600'
            }),
            dcc.Graph(id='profit-margin')
        ], style={
            'background': honey_colors['card_bg'],
            'padding': '25px',
            'borderRadius': '20px',
            'boxShadow': '0 6px 20px rgba(0,0,0,0.1)',
            'border': f'1px solid {honey_colors["secondary"]}'
        })
    ], style={'margin': '0 40px 30px 40px'}),
    
    # Invoice Details Section with Toggle Button
    html.Div([
        html.Div([
            html.Div([
                html.H2("📄 Invoice Details", style={
                    'color': honey_colors['accent'],
                    'margin': '0',
                    'fontSize': '1.5rem',
                    'fontWeight': '600'
                }),
                html.Button([
                    "👁️ ",
                    html.Span("Show/Hide Details")
                ], id='toggle-table-button', style={
                    'background': f'linear-gradient(135deg, {honey_colors["success"]} 0%, #32CD32 100%)',
                    'color': 'white',
                    'padding': '10px 20px',
                    'border': 'none',
                    'borderRadius': '20px',
                    'cursor': 'pointer',
                    'fontSize': '14px',
                    'fontWeight': '500',
                    'boxShadow': '0 3px 10px rgba(34, 139, 34, 0.3)',
                    'transition': 'all 0.3s ease',
                    'display': 'flex',
                    'alignItems': 'center',
                    'gap': '6px'
                })
            ], style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center', 'marginBottom': '15px'}),
            
            # Data Table (hidden by default)
            html.Div([
                dash_table.DataTable(
                    id='invoice-table',
                    columns=[
                        {'name': 'Invoice ID', 'id': 'invoice_id'},
                        {'name': 'Date', 'id': 'invoice_date_str'},
                        {'name': 'Customer', 'id': 'customer_name'},
                        {'name': 'Location', 'id': 'customer_location'},
                        {'name': 'Product', 'id': 'product'},
                        {'name': 'Quantity', 'id': 'qty'},
                        {'name': 'Total', 'id': 'total_str'},
                        {'name': 'Status', 'id': 'payment_status'}
                    ],
                    style_table={'overflowX': 'auto'},
                    style_cell={
                        'textAlign': 'left',
                        'padding': '12px',
                        'fontFamily': 'Segoe UI, sans-serif',
                        'fontSize': '14px'
                    },
                    style_header={
                        'backgroundColor': honey_colors['primary'],
                        'color': 'white',
                        'fontWeight': '600',
                        'border': 'none'
                    },
                    style_data={
                        'backgroundColor': honey_colors['card_bg'],
                        'border': f'1px solid {honey_colors["secondary"]}'
                    },
                    page_size=10,
                    filter_action="native",
                    sort_action="native",
                )
            ], id='table-container', style={'display': 'none'})
        ], style={
            'background': honey_colors['card_bg'],
            'padding': '25px',
            'borderRadius': '20px',
            'boxShadow': '0 6px 20px rgba(0,0,0,0.1)',
            'border': f'1px solid {honey_colors["secondary"]}'
        })
    ], style={'margin': '0 40px 40px 40px'})
], style={'minHeight': '100vh', 'paddingBottom': '40px'})

# Callback to load data on page load or manual refresh
@app.callback(
    [Output('data-store', 'data'),
     Output('last-update-time', 'children'),
     Output('status-indicator', 'children')],
    [Input('refresh-button', 'n_clicks')],
    prevent_initial_call=False  # Allow initial load
)
def update_data_store(n_clicks):
    print(f"Loading data - Button clicks: {n_clicks}")
    df = load_invoice_data()
    
    # Convert DataFrame to dict for storage
    data_dict = df.to_dict('records') if not df.empty else []
    
    # Update timestamp
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Status message with honey theme
    if df.empty:
        status_msg = "🍯 No honey sales data available yet. Upload some invoices to see your sweet analytics!"
    else:
        status_msg = f"🐝 Dashboard loaded successfully! Showing {len(df)} records from {df['invoice_id'].nunique()} unique invoices."
    
    return data_dict, f"⏰ Last Updated: {current_time}", status_msg

# Callback to update date range when data changes
@app.callback(
    [Output('date-range', 'min_date_allowed'),
     Output('date-range', 'max_date_allowed'),
     Output('date-range', 'start_date'),
     Output('date-range', 'end_date')],
    [Input('data-store', 'data')]
)
def update_date_range(data):
    if not data:
        return None, None, None, None
    
    df = pd.DataFrame(data)
    if df.empty or 'invoice_date' not in df.columns:
        return None, None, None, None
    
    # Convert invoice_date back to datetime
    df['invoice_date'] = pd.to_datetime(df['invoice_date'], errors='coerce')
    
    # Filter out invalid dates
    valid_dates = df['invoice_date'].dropna()
    
    if valid_dates.empty:
        return None, None, None, None
    
    min_date = valid_dates.min()
    max_date = valid_dates.max()
    
    return min_date, max_date, min_date, max_date

# Main callback to update all dashboard components
@app.callback(
    [
        Output('total-revenue', 'children'),
        Output('invoice-count', 'children'),
        Output('avg-invoice', 'children'),
        Output('payment-rate', 'children'),
        Output('revenue-trend-graph', 'figure'),
        Output('product-distribution', 'figure'),
        Output('product-revenue', 'figure'),
        Output('location-revenue', 'figure'),
        Output('customer-type-revenue', 'figure'),
        Output('profit-margin', 'figure'),
        Output('invoice-table', 'data')
    ],
    [
        Input('data-store', 'data'),
        Input('date-range', 'start_date'),
        Input('date-range', 'end_date')
    ]
)
def update_dashboard(data, start_date, end_date):
    # Custom color palette for charts
    honey_chart_colors = [honey_colors['primary'], honey_colors['accent'], honey_colors['warning'], 
                         honey_colors['success'], '#CD853F', '#DAA520', '#B8860B', '#FFB347']
    
    # Handle empty data
    if not data:
        empty_fig = px.scatter()
        empty_fig.update_layout(
            title="🍯 No data available - Click refresh or upload invoices",
            plot_bgcolor=honey_colors['background'],
            paper_bgcolor=honey_colors['card_bg']
        )
        return ("AED 0.00", "0", "AED 0.00", "0.0%", empty_fig, empty_fig, 
                empty_fig, empty_fig, empty_fig, empty_fig, [])
    
    # Convert data back to DataFrame
    df = pd.DataFrame(data)
    
    if df.empty:
        empty_fig = px.scatter()
        empty_fig.update_layout(
            title="🍯 No data available - Click refresh or upload invoices",
            plot_bgcolor=honey_colors['background'],
            paper_bgcolor=honey_colors['card_bg']
        )
        return ("AED 0.00", "0", "AED 0.00", "0.0%", empty_fig, empty_fig, 
                empty_fig, empty_fig, empty_fig, empty_fig, [])
    
    # Convert date columns back to datetime
    date_columns = ['invoice_date', 'due_date']
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Apply date filter if dates are provided
    if start_date and end_date:
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        filtered_df = df[(df['invoice_date'] >= start_date) & (df['invoice_date'] <= end_date)]
    else:
        filtered_df = df
    
    # If filtering results in empty dataframe
    if filtered_df.empty:
        empty_fig = px.scatter()
        empty_fig.update_layout(
            title="🍯 No data available for selected date range",
            plot_bgcolor=honey_colors['background'],
            paper_bgcolor=honey_colors['card_bg']
        )
        return ("AED 0.00", "0", "AED 0.00", "0.0%", empty_fig, empty_fig, 
                empty_fig, empty_fig, empty_fig, empty_fig, [])
    
    # Calculate KPIs
    total_revenue = f"AED {filtered_df['total'].sum():,.2f}"
    invoice_count = len(filtered_df['invoice_id'].unique())
    
    # Calculate average invoice value
    invoice_totals = filtered_df.groupby('invoice_id')['total'].sum()
    avg_invoice = f"AED {invoice_totals.mean():,.2f}" if not invoice_totals.empty else "AED 0.00"
    
    # Calculate payment rate 
    paid_amount = filtered_df[filtered_df['payment_status'] == 'Paid']['total'].sum()
    total_amount = filtered_df['total'].sum()
    payment_rate = f"{(paid_amount / total_amount * 100) if total_amount > 0 else 0:.1f}%"
    
    # Revenue Trend Graph with honey styling
    df_monthly = filtered_df.copy()
    df_monthly['month'] = df_monthly['invoice_date'].dt.strftime('%Y-%m')
    
    # Handle case when amount_excl_vat column doesn't exist
    if 'amount_excl_vat' not in df_monthly.columns:
        df_monthly['amount_excl_vat'] = df_monthly['total'] / 1.05  # Assuming 5% VAT
    
    # Handle case when vat column doesn't exist
    if 'vat' not in df_monthly.columns:
        df_monthly['vat'] = df_monthly['total'] - df_monthly['amount_excl_vat']
    
    monthly_revenue = df_monthly.groupby('month').agg(
        revenue=('total', 'sum'),
        vat=('vat', 'sum'),
        amount_excl_vat=('amount_excl_vat', 'sum')
    ).reset_index()
    
    revenue_trend = go.Figure()
    revenue_trend.add_trace(go.Bar(
        x=monthly_revenue['month'],
        y=monthly_revenue['amount_excl_vat'],
        name='Base Amount',
        marker_color=honey_colors['primary']
    ))
    revenue_trend.add_trace(go.Bar(
        x=monthly_revenue['month'],
        y=monthly_revenue['vat'],
        name='VAT',
        marker_color=honey_colors['warning']
    ))
    revenue_trend.update_layout(
        barmode='stack',
        title='Monthly Revenue Trend',
        xaxis_title='Month',
        yaxis_title='Revenue (AED)',
        plot_bgcolor=honey_colors['background'],
        paper_bgcolor=honey_colors['card_bg'],
        font=dict(color=honey_colors['text_dark']),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Product Distribution with honey styling
    if 'product' in filtered_df.columns:
        product_dist = px.pie(
            filtered_df, 
            names='product', 
            values='qty',
            title='Product Quantity Distribution',
            color_discrete_sequence=honey_chart_colors
        )
        product_dist.update_layout(
            plot_bgcolor=honey_colors['background'],
            paper_bgcolor=honey_colors['card_bg'],
            font=dict(color=honey_colors['text_dark'])
        )
    else:
        product_dist = px.pie(title='Product data not available')
        product_dist.update_layout(
            plot_bgcolor=honey_colors['background'],
            paper_bgcolor=honey_colors['card_bg']
        )
    
    # Product Revenue with honey styling
    if 'product' in filtered_df.columns:
        product_revenue = filtered_df.groupby('product')['total'].sum().reset_index()
        product_revenue = product_revenue.sort_values('total', ascending=False)
        
        product_rev_fig = px.bar(
            product_revenue,
            x='product',
            y='total',
            title='Revenue by Product',
            color='total',
            color_continuous_scale=[[0, honey_colors['secondary']], [1, honey_colors['accent']]]
        )
        product_rev_fig.update_layout(
            plot_bgcolor=honey_colors['background'],
            paper_bgcolor=honey_colors['card_bg'],
            font=dict(color=honey_colors['text_dark'])
        )
    else:
        product_rev_fig = px.bar(title='Product revenue data not available')
        product_rev_fig.update_layout(
            plot_bgcolor=honey_colors['background'],
            paper_bgcolor=honey_colors['card_bg']
        )
    
    # Location Revenue with honey styling
    if 'customer_location' in filtered_df.columns:
        location_revenue = filtered_df.groupby('customer_location')['total'].sum().reset_index()
        location_revenue = location_revenue.sort_values('total', ascending=False)
        
        location_fig = px.bar(
            location_revenue,
            x='customer_location',
            y='total',
            title='Revenue by Location',
            color='total',
            color_continuous_scale=[[0, honey_colors['secondary']], [1, honey_colors['primary']]]
        )
        location_fig.update_layout(
            plot_bgcolor=honey_colors['background'],
            paper_bgcolor=honey_colors['card_bg'],
            font=dict(color=honey_colors['text_dark'])
        )
    else:
        location_fig = px.bar(title='Location data not available')
        location_fig.update_layout(
            plot_bgcolor=honey_colors['background'],
            paper_bgcolor=honey_colors['card_bg']
        )
    
    # Customer Type Revenue with honey styling
    if 'customer_type' in filtered_df.columns:
        type_revenue = filtered_df.groupby('customer_type')['total'].sum().reset_index()
        type_revenue = type_revenue.sort_values('total', ascending=False)
        
        type_fig = px.pie(
            type_revenue,
            names='customer_type',
            values='total',
            title='Revenue by Customer Type',
            color_discrete_sequence=honey_chart_colors
        )
        type_fig.update_layout(
            plot_bgcolor=honey_colors['background'],
            paper_bgcolor=honey_colors['card_bg'],
            font=dict(color=honey_colors['text_dark'])
        )
    else:
        type_fig = px.pie(
            pd.DataFrame({'type': ['Unknown'], 'value': [1]}),
            names='type',
            values='value',
            title='Customer Type Data Not Available'
        )
        type_fig.update_layout(
            plot_bgcolor=honey_colors['background'],
            paper_bgcolor=honey_colors['card_bg']
        )
    
    # Profit Margin Analysis with honey styling
    if 'profit' in filtered_df.columns and 'amount_excl_vat' in filtered_df.columns:
        df_profit = filtered_df.copy()
        df_profit['month'] = df_profit['invoice_date'].dt.strftime('%Y-%m')
        profit_data = df_profit.groupby('month').agg(
            revenue=('amount_excl_vat', 'sum'),
            profit=('profit', 'sum')
        ).reset_index()
        profit_data['margin'] = profit_data['profit'] / profit_data['revenue'] * 100
        
        profit_fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        profit_fig.add_trace(
            go.Bar(x=profit_data['month'], y=profit_data['profit'], name="Profit", 
                   marker_color=honey_colors['success']),
            secondary_y=False
        )
        
        profit_fig.add_trace(
            go.Scatter(x=profit_data['month'], y=profit_data['margin'], name="Margin %", 
                      line=dict(color=honey_colors['accent'], width=3)),
            secondary_y=True
        )
        
        profit_fig.update_layout(
            title_text="Monthly Profit and Margin",
            plot_bgcolor=honey_colors['background'],
            paper_bgcolor=honey_colors['card_bg'],
            font=dict(color=honey_colors['text_dark']),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        profit_fig.update_yaxes(title_text="Profit (AED)", secondary_y=False)
        profit_fig.update_yaxes(title_text="Margin (%)", secondary_y=True)
    else:
        profit_fig = px.scatter(title="Profit Data Not Available")
        profit_fig.update_layout(
            plot_bgcolor=honey_colors['background'],
            paper_bgcolor=honey_colors['card_bg']
        )
    
    # Data Table
    table_df = filtered_df.copy()
    
    # Prepare table data with safe column access
    table_data = []
    for _, row in table_df.iterrows():
        table_row = {
            'invoice_id': row.get('invoice_id', 'N/A'),
            'invoice_date_str': row['invoice_date'].strftime('%Y-%m-%d') if pd.notna(row.get('invoice_date')) else 'N/A',
            'customer_name': row.get('customer_name', 'N/A'),
            'customer_location': row.get('customer_location', 'N/A'),
            'product': row.get('product', 'N/A'),
            'qty': row.get('qty', 0),
            'total_str': f"AED {row.get('total', 0):,.2f}",
            'payment_status': row.get('payment_status', 'N/A')
        }
        table_data.append(table_row)
    
    return total_revenue, invoice_count, avg_invoice, payment_rate, revenue_trend, product_dist, product_rev_fig, location_fig, type_fig, profit_fig, table_data

# Toggle table visibility callback
@app.callback(
    Output('table-container', 'style'),
    [Input('toggle-table-button', 'n_clicks')],
    [State('table-container', 'style')]
)
def toggle_table(n_clicks, current_style):
    if n_clicks is None:
        return {'display': 'none'}
    
    if current_style is None or 'display' not in current_style:
        current_style = {'display': 'none'}
    
    if current_style.get('display') == 'none':
        return {'display': 'block'}
    else:
        return {'display': 'none'}

