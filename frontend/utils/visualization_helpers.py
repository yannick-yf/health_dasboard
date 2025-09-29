"""
Helper functions for creating visualizations
"""

import numpy as np
import plotly.graph_objects as go


def create_time_series_plot(filtered_df, metric, aggregation):
    """Create enhanced time series plot for a metric"""
    fig = go.Figure()
    
    # Metric configuration
    metric_config = {
        'steps': {'unit': 'steps', 'color': '#1f77b4'},
        'sleep_min': {'unit': 'minutes', 'color': '#9467bd'},
        'workout_duration_min_tot': {'unit': 'minutes', 'color': '#ff7f0e'},
        'weight': {'unit': 'kg', 'color': '#d62728'},
        'calories_burned': {'unit': 'kcal', 'color': '#ff6b6b'},
        'calories_consumed': {'unit': 'kcal', 'color': '#4ecdc4'}
    }
    
    config = metric_config.get(metric, {'unit': '', 'color': '#2ca02c'})
    
    # Main data line
    fig.add_trace(go.Scatter(
        x=filtered_df['date'],
        y=filtered_df[metric],
        mode='lines+markers',
        name=metric.replace('_', ' ').title(),
        line=dict(width=3, color=config['color']),
        marker=dict(size=6),
        hovertemplate='<b>%{y:.1f}</b> ' + config['unit'] + '<br>%{x}<extra></extra>'
    ))
    
    # Add moving averages for daily data
    if aggregation == "Daily" and len(filtered_df) > 7:
        # 7-day moving average
        ma7 = filtered_df[metric].rolling(window=7, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=filtered_df['date'],
            y=ma7,
            mode='lines',
            name='7-day Average',
            line=dict(dash='dash', width=2, color=config['color']),
            opacity=0.7,
            hovertemplate='7-day avg: <b>%{y:.1f}</b> ' + config['unit'] + '<extra></extra>'
        ))
        
        # 30-day moving average for longer periods
        if len(filtered_df) > 30:
            ma30 = filtered_df[metric].rolling(window=30, min_periods=1).mean()
            fig.add_trace(go.Scatter(
                x=filtered_df['date'],
                y=ma30,
                mode='lines',
                name='30-day Average',
                line=dict(dash='dot', width=2, color=config['color']),
                opacity=0.5,
                hovertemplate='30-day avg: <b>%{y:.1f}</b> ' + config['unit'] + '<extra></extra>'
            ))
    
    # Add trend line if we have enough data points
    if len(filtered_df[filtered_df[metric].notna()]) >= 3:
        trend_data = filtered_df[filtered_df[metric].notna()].reset_index(drop=True)
        if len(trend_data) > 1:
            x_numeric = np.arange(len(trend_data))
            z = np.polyfit(x_numeric, trend_data[metric], 1)
            p = np.poly1d(z)
            
            # Determine trend direction for color
            trend_color = '#28a745' if z[0] > 0 else '#dc3545' if z[0] < 0 else '#6c757d'
            
            fig.add_trace(go.Scatter(
                x=trend_data['date'],
                y=p(x_numeric),
                mode='lines',
                name='Trend',
                line=dict(dash='dashdot', width=2, color=trend_color),
                opacity=0.8,
                hovertemplate='Trend: <b>%{y:.1f}</b> ' + config['unit'] + '<extra></extra>'
            ))
    
    # Enhanced layout
    fig.update_layout(
        title={
            'text': f"{metric.replace('_', ' ').title()} Over Time ({aggregation} View)",
            'x': 0.5,
            'font': {'size': 18}
        },
        xaxis_title="Date",
        yaxis_title=f"{metric.replace('_', ' ').title()} ({config['unit']})",
        height=450,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    
    # Add grid
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', griddash='dot')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', griddash='dot')
    
    return fig