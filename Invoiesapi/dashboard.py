import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random
from datetime import datetime, timedelta
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Global variable to track data version
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

# Create the Dash app
app = dash.Dash(__name__, title="Invoice Analytics Dashboard")

# Define the layout
app.layout = html.Div([
    # Manual refresh button
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
    
    html.H1("Invoice Analytics Dashboard", style={'textAlign': 'center'}),
    
    # Status indicator
    html.Div([
        html.Div(id='status-indicator', style={
            'textAlign': 'center',
            'padding': '10px',
            'margin': '10px',
            'borderRadius': '5px',
            'backgroundColor': '#e8f5e8',
            'border': '1px solid #4CAF50'
        })
    ]),
    
    # Date Range Filter
    html.Div([
        html.H4("Filter by Date Range:"),
        dcc.DatePickerRange(
            id='date-range',
            display_format='YYYY-MM-DD'
        ),
    ], style={'margin': '20px'}),
    
    # KPI Cards
    html.Div([
        html.Div([
            html.H3("Total Revenue"),
            html.H2(id='total-revenue')
        ], className='card', style={'border': '1px solid #ddd', 'padding': '20px', 'margin': '10px', 'borderRadius': '5px', 'textAlign': 'center'}),
        html.Div([
            html.H3("Invoices Count"),
            html.H2(id='invoice-count')
        ], className='card', style={'border': '1px solid #ddd', 'padding': '20px', 'margin': '10px', 'borderRadius': '5px', 'textAlign': 'center'}),
        html.Div([
            html.H3("Average Invoice Value"),
            html.H2(id='avg-invoice')
        ], className='card', style={'border': '1px solid #ddd', 'padding': '20px', 'margin': '10px', 'borderRadius': '5px', 'textAlign': 'center'}),
        html.Div([
            html.H3("Payment Rate"),
            html.H2(id='payment-rate')
        ], className='card', style={'border': '1px solid #ddd', 'padding': '20px', 'margin': '10px', 'borderRadius': '5px', 'textAlign': 'center'})
    ], style={'display': 'flex', 'justifyContent': 'space-between', 'margin': '20px'}),
    
    # Revenue Trends
    html.Div([
        html.H2("Revenue Trends"),
        dcc.Graph(id='revenue-trend-graph')
    ]),
    
    # Product Performance
    html.Div([
        html.Div([
            html.H2("Product Sales Distribution"),
            dcc.Graph(id='product-distribution')
        ], style={'width': '48%'}),
        html.Div([
            html.H2("Top Products by Revenue"),
            dcc.Graph(id='product-revenue')
        ], style={'width': '48%'})
    ], style={'display': 'flex', 'justifyContent': 'space-between'}),
    
    # Customer Insights
    html.Div([
        html.H2("Customer Insights"),
        html.Div([
            html.Div([
                html.H3("Revenue by Customer Location"),
                dcc.Graph(id='location-revenue')
            ], style={'width': '48%'}),
            html.Div([
                html.H3("Revenue by Customer Type"),
                dcc.Graph(id='customer-type-revenue')
            ], style={'width': '48%'})
        ], style={'display': 'flex', 'justifyContent': 'space-between'})
    ]),
    
    # Financial Analysis
    html.Div([
        html.H2("Financial Analysis"),
        html.Div([
            html.H3("Profit Margin Analysis"),
            dcc.Graph(id='profit-margin')
        ])
    ]),
    
    # Invoice Details Section with Toggle Button
    html.Div([
        html.Div([
            html.H2("Invoice Details", style={'display': 'inline-block', 'marginRight': '20px'}),
            html.Button(
                'Show/Hide Details', 
                id='toggle-table-button',
                style={
                    'backgroundColor': '#4CAF50',
                    'color': 'white',
                    'padding': '10px 15px',
                    'border': 'none',
                    'borderRadius': '4px',
                    'cursor': 'pointer',
                    'fontSize': '16px'
                }
            )
        ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '15px'}),
        
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
                    'padding': '10px'
                },
                style_header={
                    'backgroundColor': 'rgb(230, 230, 230)',
                    'fontWeight': 'bold'
                },
                page_size=10,
                filter_action="native",
                sort_action="native",
            )
        ], id='table-container', style={'display': 'none'})
    ])
])

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
    
    # Status message
    if df.empty:
        status_msg = "📊 No invoice data available. Upload some invoices to see analytics."
    else:
        status_msg = f"📈 Dashboard loaded! Showing {len(df)} records from {df['invoice_id'].nunique()} unique invoices."
    
    return data_dict, f"Last Updated: {current_time}", status_msg

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
    # Handle empty data
    if not data:
        empty_fig = px.scatter()
        empty_fig.update_layout(title="No data available - Click refresh or upload invoices")
        return ("AED 0.00", "0", "AED 0.00", "0.0%", empty_fig, empty_fig, 
                empty_fig, empty_fig, empty_fig, empty_fig, [])
    
    # Convert data back to DataFrame
    df = pd.DataFrame(data)
    
    if df.empty:
        empty_fig = px.scatter()
        empty_fig.update_layout(title="No data available - Click refresh or upload invoices")
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
        empty_fig.update_layout(title="No data available for selected date range")
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
    
    # Revenue Trend Graph
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
        name='Base Amount'
    ))
    revenue_trend.add_trace(go.Bar(
        x=monthly_revenue['month'],
        y=monthly_revenue['vat'],
        name='VAT'
    ))
    revenue_trend.update_layout(
        barmode='stack',
        title='Monthly Revenue Trend',
        xaxis_title='Month',
        yaxis_title='Revenue (AED)',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Product Distribution
    if 'product' in filtered_df.columns:
        product_dist = px.pie(
            filtered_df, 
            names='product', 
            values='qty',
            title='Product Quantity Distribution'
        )
    else:
        product_dist = px.pie(title='Product data not available')
    
    # Product Revenue
    if 'product' in filtered_df.columns:
        product_revenue = filtered_df.groupby('product')['total'].sum().reset_index()
        product_revenue = product_revenue.sort_values('total', ascending=False)
        
        product_rev_fig = px.bar(
            product_revenue,
            x='product',
            y='total',
            title='Revenue by Product'
        )
    else:
        product_rev_fig = px.bar(title='Product revenue data not available')
    
    # Location Revenue
    if 'customer_location' in filtered_df.columns:
        location_revenue = filtered_df.groupby('customer_location')['total'].sum().reset_index()
        location_revenue = location_revenue.sort_values('total', ascending=False)
        
        location_fig = px.bar(
            location_revenue,
            x='customer_location',
            y='total',
            title='Revenue by Location'
        )
    else:
        location_fig = px.bar(title='Location data not available')
    
    # Customer Type Revenue
    if 'customer_type' in filtered_df.columns:
        type_revenue = filtered_df.groupby('customer_type')['total'].sum().reset_index()
        type_revenue = type_revenue.sort_values('total', ascending=False)
        
        type_fig = px.pie(
            type_revenue,
            names='customer_type',
            values='total',
            title='Revenue by Customer Type'
        )
    else:
        type_fig = px.pie(
            pd.DataFrame({'type': ['Unknown'], 'value': [1]}),
            names='type',
            values='value',
            title='Customer Type Data Not Available'
        )
    
    # Profit Margin Analysis
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
            go.Bar(x=profit_data['month'], y=profit_data['profit'], name="Profit"),
            secondary_y=False
        )
        
        profit_fig.add_trace(
            go.Scatter(x=profit_data['month'], y=profit_data['margin'], name="Margin %", line=dict(color='red')),
            secondary_y=True
        )
        
        profit_fig.update_layout(
            title_text="Monthly Profit and Margin",
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

# Run the app
if __name__ == '__main__':
    app.run(debug=True)