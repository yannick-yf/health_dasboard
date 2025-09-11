"""
Visualization helper functions for Health Tracker application
Creates interactive plots and dashboards using Plotly
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, List, Dict, Any


def create_time_series_plot(df: pd.DataFrame, 
                           column: str, 
                           title: str = None,
                           show_ma: bool = True,
                           ma_windows: List[int] = [7, 30]) -> go.Figure:
    """
    Create an interactive time series plot with optional moving averages
    
    Args:
        df: DataFrame with date column and metric
        column: Column name to plot
        title: Plot title
        show_ma: Whether to show moving averages
        ma_windows: List of moving average windows
    
    Returns:
        Plotly figure object
    """
    if title is None:
        title = f"{column.replace('_', ' ').title()} Over Time"
    
    fig = go.Figure()
    
    # Main line
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df[column],
        mode='lines+markers',
        name=column.replace('_', ' ').title(),
        line=dict(width=2),
        marker=dict(size=6),
        hovertemplate='%{y:.1f}<extra></extra>'
    ))
    
    # Add moving averages
    if show_ma and len(df) > min(ma_windows):
        colors = ['orange', 'red', 'green']
        for i, window in enumerate(ma_windows):
            if len(df) >= window:
                ma = df[column].rolling(window=window, min_periods=1).mean()
                fig.add_trace(go.Scatter(
                    x=df['date'],
                    y=ma,
                    mode='lines',
                    name=f'{window}-day MA',
                    line=dict(dash='dash', width=1, color=colors[i % len(colors)]),
                    hovertemplate='%{y:.1f}<extra></extra>'
                ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title=column.replace('_', ' ').title(),
        hovermode='x unified',
        showlegend=True,
        height=400,
        template='plotly_white'
    )
    
    return fig


def create_correlation_heatmap(df: pd.DataFrame, 
                              columns: Optional[List[str]] = None) -> go.Figure:
    """
    Create correlation heatmap for numeric columns
    
    Args:
        df: DataFrame with numeric columns
        columns: List of columns to include (None for all numeric)
    
    Returns:
        Plotly figure object
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Calculate correlation matrix
    corr_matrix = df[columns].corr()
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu_r',
        zmid=0,
        zmin=-1,
        zmax=1,
        text=corr_matrix.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(
        title="Correlation Matrix",
        height=500,
        xaxis=dict(tickangle=45),
        template='plotly_white'
    )
    
    return fig


def create_distribution_plot(series: pd.Series, 
                            title: str = None,
                            nbins: int = 20) -> go.Figure:
    """
    Create distribution plot with histogram and box plot
    
    Args:
        series: Data series to plot
        title: Plot title
        nbins: Number of histogram bins
    
    Returns:
        Plotly figure object
    """
    if title is None:
        title = f"Distribution of {series.name}"
    
    # Create figure with secondary y-axis
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.05,
        subplot_titles=(title, "")
    )
    
    # Histogram
    fig.add_trace(
        go.Histogram(
            x=series.dropna(),
            nbinsx=nbins,
            name='Frequency',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Box plot
    fig.add_trace(
        go.Box(
            x=series.dropna(),
            name='',
            showlegend=False,
            marker_color='lightblue'
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        height=400,
        showlegend=False,
        template='plotly_white'
    )
    
    fig.update_xaxes(title_text=series.name, row=2, col=1)
    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    
    return fig


def create_kpi_cards(metrics: Dict[str, Any]) -> str:
    """
    Create HTML for KPI cards
    
    Args:
        metrics: Dictionary of metric names and values
    
    Returns:
        HTML string for KPI cards
    """
    html = '<div style="display: flex; justify-content: space-around; flex-wrap: wrap;">'
    
    for name, value in metrics.items():
        if pd.isna(value):
            value_str = "N/A"
        elif isinstance(value, (int, float)):
            if value >= 1000:
                value_str = f"{value:,.0f}"
            else:
                value_str = f"{value:.1f}"
        else:
            value_str = str(value)
        
        html += f'''
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin: 10px;
            min-width: 150px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        ">
            <div style="font-size: 24px; font-weight: bold;">{value_str}</div>
            <div style="font-size: 14px; margin-top: 5px;">{name}</div>
        </div>
        '''
    
    html += '</div>'
    return html


def create_comparison_plot(df: pd.DataFrame,
                          columns: List[str],
                          title: str = "Metric Comparison") -> go.Figure:
    """
    Create a comparison plot for multiple metrics
    
    Args:
        df: DataFrame with date and metric columns
        columns: List of column names to compare
        title: Plot title
    
    Returns:
        Plotly figure object
    """
    fig = make_subplots(
        rows=len(columns), 
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=[col.replace('_', ' ').title() for col in columns]
    )
    
    colors = px.colors.qualitative.Set2
    
    for i, col in enumerate(columns):
        if col in df.columns and df[col].notna().any():
            fig.add_trace(
                go.Scatter(
                    x=df['date'],
                    y=df[col],
                    mode='lines+markers',
                    name=col.replace('_', ' ').title(),
                    line=dict(color=colors[i % len(colors)], width=2),
                    marker=dict(size=4),
                    showlegend=False
                ),
                row=i+1, col=1
            )
    
    fig.update_layout(
        title=title,
        height=200 * len(columns),
        hovermode='x unified',
        template='plotly_white'
    )
    
    fig.update_xaxes(title_text="Date", row=len(columns), col=1)
    
    return fig


def create_weekly_summary_plot(df: pd.DataFrame) -> go.Figure:
    """
    Create weekly summary visualization
    
    Args:
        df: DataFrame with daily health data
    
    Returns:
        Plotly figure object
    """
    # Resample to weekly
    df_weekly = df.set_index('date').resample('W').agg({
        'steps': 'sum',
        'sleep_min': 'mean',
        'workout_duration_min_tot': 'sum',
        'calories_burned': 'sum',
        'calories_consumed': 'sum'
    }).reset_index()
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Weekly Steps', 'Avg Sleep (hours)', 
                       'Weekly Workout (hours)', 'Weekly Calorie Balance'),
        vertical_spacing=0.12,
        horizontal_spacing=0.15
    )
    
    # Weekly steps
    fig.add_trace(
        go.Bar(x=df_weekly['date'], y=df_weekly['steps'], 
               marker_color='lightblue', name='Steps'),
        row=1, col=1
    )
    
    # Average sleep
    fig.add_trace(
        go.Bar(x=df_weekly['date'], y=df_weekly['sleep_min']/60,
               marker_color='purple', name='Sleep'),
        row=1, col=2
    )
    
    # Weekly workout
    fig.add_trace(
        go.Bar(x=df_weekly['date'], y=df_weekly['workout_duration_min_tot']/60,
               marker_color='green', name='Workout'),
        row=2, col=1
    )
    
    # Calorie balance
    fig.add_trace(
        go.Bar(x=df_weekly['date'], 
               y=df_weekly['calories_consumed'] - df_weekly['calories_burned'],
               marker_color='orange', name='Balance'),
        row=2, col=2
    )
    
    fig.update_layout(
        height=600,
        showlegend=False,
        template='plotly_white'
    )
    
    return fig


def create_progress_gauge(current: float, target: float, title: str) -> go.Figure:
    """
    Create a gauge chart for progress tracking
    
    Args:
        current: Current value
        target: Target value
        title: Gauge title
    
    Returns:
        Plotly figure object
    """
    percentage = min((current / target) * 100, 100) if target > 0 else 0
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=current,
        delta={'reference': target},
        title={'text': title},
        gauge={
            'axis': {'range': [None, target * 1.2]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, target * 0.5], 'color': "lightgray"},
                {'range': [target * 0.5, target * 0.8], 'color': "gray"},
                {'range': [target * 0.8, target], 'color': "lightgreen"},
                {'range': [target, target * 1.2], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': target
            }
        }
    ))
    
    fig.update_layout(height=250, template='plotly_white')
    
    return fig


def create_anomaly_plot(df: pd.DataFrame, column: str, anomalies: pd.Series) -> go.Figure:
    """
    Create plot highlighting anomalies
    
    Args:
        df: DataFrame with data
        column: Column to plot
        anomalies: Boolean series indicating anomalies
    
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    # Normal points
    normal_mask = ~anomalies
    fig.add_trace(go.Scatter(
        x=df.loc[normal_mask, 'date'],
        y=df.loc[normal_mask, column],
        mode='lines+markers',
        name='Normal',
        line=dict(color='blue', width=2),
        marker=dict(size=6)
    ))
    
    # Anomaly points
    anomaly_mask = anomalies
    if anomaly_mask.any():
        fig.add_trace(go.Scatter(
            x=df.loc[anomaly_mask, 'date'],
            y=df.loc[anomaly_mask, column],
            mode='markers',
            name='Anomaly',
            marker=dict(color='red', size=10, symbol='x')
        ))
    
    fig.update_layout(
        title=f"{column.replace('_', ' ').title()} with Anomalies",
        xaxis_title="Date",
        yaxis_title=column.replace('_', ' ').title(),
        hovermode='x unified',
        template='plotly_white',
        height=400
    )
    
    return fig


def create_monthly_heatmap(df: pd.DataFrame, column: str) -> go.Figure:
    """
    Create monthly heatmap for a metric
    
    Args:
        df: DataFrame with date and metric
        column: Column to visualize
    
    Returns:
        Plotly figure object
    """
    # Extract month and day
    df_copy = df.copy()
    df_copy['month'] = df_copy['date'].dt.strftime('%Y-%m')
    df_copy['day'] = df_copy['date'].dt.day
    
    # Pivot for heatmap
    pivot = df_copy.pivot_table(values=column, index='day', columns='month', aggfunc='mean')
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        colorscale='Viridis',
        colorbar=dict(title=column.replace('_', ' ').title())
    ))
    
    fig.update_layout(
        title=f"Monthly Heatmap: {column.replace('_', ' ').title()}",
        xaxis_title="Month",
        yaxis_title="Day of Month",
        height=500,
        template='plotly_white'
    )
    
    return fig