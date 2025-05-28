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

# Load data from the CSV file instead of generating it
def load_invoice_data():
    """Load invoice data from CSV file for dashboard visualization"""
    try:
        # Load the CSV file
        df = pd.read_csv('invoice_data.csv')
        
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

# Load the data from CSV
df = load_invoice_data()

# Create the Dash app
app = dash.Dash(__name__, title="Invoice Analytics Dashboard")

# Define the layout
app.layout = html.Div([
    html.H1("Invoice Analytics Dashboard", style={'textAlign': 'center'}),
    
    # Date Range Filter
    html.Div([
        html.H4("Filter by Date Range:"),
        dcc.DatePickerRange(
            id='date-range',
            min_date_allowed=df['invoice_date'].min() if not df.empty else None,
            max_date_allowed=df['invoice_date'].max() if not df.empty else None,
            start_date=df['invoice_date'].min() if not df.empty else None,
            end_date=df['invoice_date'].max() if not df.empty else None
        ),
    ], style={'margin': '20px'}),
    
    # KPI Cards
    html.Div([
        html.Div([
            html.H3("Total Revenue"),
            html.H2(id='total-revenue')
        ], className='card'),
        html.Div([
            html.H3("Invoices Count"),
            html.H2(id='invoice-count')
        ], className='card'),
        html.Div([
            html.H3("Average Invoice Value"),
            html.H2(id='avg-invoice')
        ], className='card'),
        html.Div([
            html.H3("Payment Rate"),
            html.H2(id='payment-rate')
        ], className='card')
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
    
    # Financial Analysis - Removed Payment Timeline, only showing Profit Margin now
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
                    {'name': 'Location', 'id': 'customer_location'},  # Added customer location
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
                page_size=10,  # Add pagination for better performance
                filter_action="native",  # Enable filtering
                sort_action="native",  # Enable sorting
            )
        ], id='table-container', style={'display': 'none'})  # Hidden by default
    ])
])

# Callbacks
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
        Input('date-range', 'start_date'),
        Input('date-range', 'end_date')
    ]
)
def update_dashboard(start_date, end_date):
    # Handle empty dataframe or invalid date range
    if df.empty or not start_date or not end_date:
        empty_fig = px.scatter()
        empty_fig.update_layout(title="No data available")
        return ("AED 0.00", "0", "AED 0.00", "0.0%", empty_fig, empty_fig, 
                empty_fig, empty_fig, empty_fig, empty_fig, [])
    
    # Convert string dates to datetime if needed
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)
    
    # Filter data by date range
    filtered_df = df[(df['invoice_date'] >= start_date) & (df['invoice_date'] <= end_date)]
    
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
    product_dist = px.pie(
        filtered_df, 
        names='product', 
        values='qty',
        title='Product Quantity Distribution'
    )
    
    # Product Revenue
    product_revenue = filtered_df.groupby('product')['total'].sum().reset_index()
    product_revenue = product_revenue.sort_values('total', ascending=False)
    
    product_rev_fig = px.bar(
        product_revenue,
        x='product',
        y='total',
        title='Revenue by Product'
    )
    
    # Location Revenue
    location_revenue = filtered_df.groupby('customer_location')['total'].sum().reset_index()
    location_revenue = location_revenue.sort_values('total', ascending=False)
    
    location_fig = px.bar(
        location_revenue,
        x='customer_location',
        y='total',
        title='Revenue by Location'
    )
    
    # Customer Type Revenue - handle missing customer_type column
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
    table_df['invoice_date_str'] = table_df['invoice_date'].dt.strftime('%Y-%m-%d')
    table_df['total_str'] = table_df['total'].apply(lambda x: f"AED {x:,.2f}")
    
    # Ensure all required columns exist
    required_cols = ['invoice_id', 'invoice_date_str', 'customer_name', 'customer_location', 'product', 'qty', 'total_str', 'payment_status']
    for col in required_cols:
        if col not in table_df.columns:
            if col == 'invoice_date_str':  # Special case for formatted date
                continue
            elif col == 'total_str':  # Special case for formatted total
                continue
            else:
                table_df[col] = "N/A"
    
    table_data = table_df[required_cols].to_dict('records')
    
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

