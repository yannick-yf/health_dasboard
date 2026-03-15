"""
Helper functions for creating visualizations
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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


_DARK_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font_color='#f0f2f6',
    height=380,
    hovermode='x unified',
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    xaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)', griddash='dot'),
    yaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)', griddash='dot'),
)


def create_weight_chart(df):
    """Weight line + 7-day MA + regression + 76 kg target reference."""
    data = df[df['weight'].notna()].copy()
    if data.empty:
        return go.Figure()

    ma7 = data['weight'].rolling(window=7, min_periods=1).mean()

    # Linear regression
    x_num = np.arange(len(data))
    z = np.polyfit(x_num, data['weight'], 1)
    p = np.poly1d(z)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data['date'], y=data['weight'],
        mode='lines+markers', name='Weight',
        line=dict(color='#667eea', width=2), marker=dict(size=4),
        hovertemplate='%{y:.2f} kg<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=data['date'], y=ma7,
        mode='lines', name='7-day MA',
        line=dict(color='#a78bfa', width=2, dash='dash'),
        hovertemplate='MA: %{y:.2f} kg<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=data['date'], y=p(x_num),
        mode='lines', name='Trend',
        line=dict(color='#f59e0b', width=1.5, dash='dashdot'),
        hovertemplate='Trend: %{y:.2f} kg<extra></extra>'
    ))
    # Target reference line
    fig.add_hline(y=76, line_dash='dot', line_color='#10b981',
                  annotation_text='Target 76 kg', annotation_position='top left',
                  annotation_font_color='#10b981')
    # Bulk start annotation (~Nov 5 2025)
    bulk_start_ts = pd.Timestamp('2025-11-05').timestamp() * 1000
    fig.add_vline(x=bulk_start_ts, line_dash='dot', line_color='rgba(255,255,255,0.3)',
                  annotation_text='Bulk start', annotation_position='top right',
                  annotation_font_color='rgba(255,255,255,0.6)')

    layout = dict(_DARK_LAYOUT)
    layout['title'] = dict(text='Weight Progress', x=0.5)
    layout['yaxis']['title'] = 'Weight (kg)'
    fig.update_layout(**layout)
    return fig


def create_energy_balance_chart(df):
    """Grouped bars for calories consumed/burned + surplus line on secondary Y axis."""
    surplus = df['calories_consumed'] - df['calories_burned']

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Bar(
        x=df['date'], y=df['calories_consumed'],
        name='Consumed', marker_color='#14b8a6',
        hovertemplate='Consumed: %{y:.0f} kcal<extra></extra>'
    ), secondary_y=False)
    fig.add_trace(go.Bar(
        x=df['date'], y=df['calories_burned'],
        name='Burned', marker_color='#f97316',
        hovertemplate='Burned: %{y:.0f} kcal<extra></extra>'
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=df['date'], y=surplus,
        mode='lines+markers', name='Surplus/Deficit',
        line=dict(width=2),
        marker=dict(color=['#10b981' if v >= 0 else '#ef4444' for v in surplus], size=5),
        hovertemplate='Balance: %{y:.0f} kcal<extra></extra>'
    ), secondary_y=True)

    fig.add_hline(y=0, secondary_y=True, line_dash='dot', line_color='rgba(255,255,255,0.4)')

    layout = dict(_DARK_LAYOUT)
    layout['barmode'] = 'group'
    layout['title'] = dict(text='Energy Balance', x=0.5)
    fig.update_layout(**layout)
    fig.update_yaxes(title_text='Calories (kcal)', secondary_y=False)
    fig.update_yaxes(showgrid=False, title_text='Surplus (kcal)', secondary_y=True)
    return fig


def create_steps_workout_chart(df):
    """Step bars (primary Y) + workout duration line (secondary Y)."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=df['date'], y=df['steps'],
        name='Steps', marker_color='#3b82f6',
        hovertemplate='Steps: %{y:,.0f}<extra></extra>'
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['workout_duration_min_tot'],
        mode='lines+markers', name='Workout (min)',
        line=dict(color='#f97316', width=2), marker=dict(size=5),
        hovertemplate='Workout: %{y:.0f} min<extra></extra>'
    ), secondary_y=True)

    layout = dict(_DARK_LAYOUT)
    layout['title'] = dict(text='Steps & Workout', x=0.5)
    fig.update_layout(**layout)
    fig.update_yaxes(title_text='Steps', secondary_y=False)
    fig.update_yaxes(showgrid=False, title_text='Workout (min)', secondary_y=True)
    return fig


def create_sleep_chart(df):
    """Sleep bars colored by duration + 7-day MA + 7.5h reference line."""
    data = df[df['sleep_min'].notna()].copy()
    if data.empty:
        return go.Figure()

    hours = data['sleep_min'] / 60
    colors = ['#10b981' if h >= 7.5 else '#f59e0b' if h >= 7.0 else '#ef4444' for h in hours]
    ma7 = hours.rolling(window=7, min_periods=1).mean()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=data['date'], y=hours,
        name='Sleep (h)', marker_color=colors,
        hovertemplate='Sleep: %{y:.1f} h<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=data['date'], y=ma7,
        mode='lines', name='7-day MA',
        line=dict(color='#a78bfa', width=2, dash='dash'),
        hovertemplate='MA: %{y:.1f} h<extra></extra>'
    ))
    fig.add_hline(y=7.5, line_dash='dot', line_color='#10b981',
                  annotation_text='Target 7.5h', annotation_position='top left',
                  annotation_font_color='#10b981')

    layout = dict(_DARK_LAYOUT)
    layout['title'] = dict(text='Sleep Duration', x=0.5)
    layout['yaxis']['title'] = 'Hours'
    fig.update_layout(**layout)
    return fig