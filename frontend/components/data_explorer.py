"""
Comprehensive Data Explorer Component for LLM Survey Pipeline
Provides advanced filtering, visualization, and analysis tools for survey results
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import scipy.stats as stats
from pathlib import Path
import io
import base64


class DataExplorer:
    """Advanced data explorer for analyzing survey results"""
    
    def __init__(self):
        """Initialize the data explorer with default settings"""
        self.data_dir = Path("data/survey_runs")
        self.initialize_session_state()
        
    def initialize_session_state(self):
        """Initialize session state variables for the explorer"""
        if 'selected_runs' not in st.session_state:
            st.session_state.selected_runs = []
        if 'filter_settings' not in st.session_state:
            st.session_state.filter_settings = {
                'models': [],
                'scales': [],
                'prompts': [],
                'date_range': None,
                'response_type': 'all',
                'score_range': None
            }
        if 'comparison_data' not in st.session_state:
            st.session_state.comparison_data = None
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = {}
            
    def render(self):
        """Main render method for the data explorer interface"""
        st.title("Survey Data Explorer")
        st.markdown("Analyze and visualize survey results with advanced filtering and comparison tools")
        
        # Create tabs for different sections
        tabs = st.tabs([
            "Data Selection", 
            "Filtering & Query", 
            "Data Views", 
            "Visualizations", 
            "Statistical Analysis", 
            "Export & Share"
        ])
        
        with tabs[0]:
            self.render_data_selection()
            
        with tabs[1]:
            self.render_filtering_interface()
            
        with tabs[2]:
            self.render_data_views()
            
        with tabs[3]:
            self.render_visualizations()
            
        with tabs[4]:
            self.render_statistical_analysis()
            
        with tabs[5]:
            self.render_export_options()
            
    def render_data_selection(self):
        """Render the data selection interface"""
        st.header("Select Survey Runs")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Load available runs
            available_runs = self.load_available_runs()
            
            if not available_runs:
                st.warning("No survey runs found. Please run a survey first.")
                return
                
            # Quick filters
            st.subheader("Quick Filters")
            filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)
            
            with filter_col1:
                if st.button("Last 7 Days"):
                    self.apply_date_filter(7)
                    
            with filter_col2:
                if st.button("Last 30 Days"):
                    self.apply_date_filter(30)
                    
            with filter_col3:
                if st.button("This Month"):
                    self.apply_current_month_filter()
                    
            with filter_col4:
                if st.button("All Time"):
                    st.session_state.filter_settings['date_range'] = None
                    
            # Run selector with metadata display
            st.subheader("Available Survey Runs")
            
            # Create DataFrame for display
            runs_df = self.create_runs_dataframe(available_runs)
            
            # Multi-select for runs
            selected_runs = st.multiselect(
                "Select runs to analyze (compare multiple runs):",
                options=list(runs_df.index),
                default=st.session_state.selected_runs,
                format_func=lambda x: f"{x} - {runs_df.loc[x, 'Date']} ({runs_df.loc[x, 'Models']})",
                help="Select one or more survey runs to analyze and compare"
            )
            
            st.session_state.selected_runs = selected_runs
            
            # Display selected runs metadata
            if selected_runs:
                st.subheader("Selected Runs Information")
                display_df = runs_df.loc[selected_runs]
                st.dataframe(display_df, use_container_width=True)
                
        with col2:
            # Search functionality
            st.subheader("Search Runs")
            search_term = st.text_input(
                "Search by Run ID or Tags:",
                placeholder="Enter search term...",
                help="Search for specific runs by ID or associated tags"
            )
            
            if search_term:
                filtered_runs = self.search_runs(available_runs, search_term)
                if filtered_runs:
                    st.success(f"Found {len(filtered_runs)} matching runs")
                    for run_id in filtered_runs[:5]:  # Show first 5 matches
                        st.write(f"- {run_id}")
                else:
                    st.info("No matching runs found")
                    
            # Load selected data
            if st.button("Load Selected Data", type="primary", use_container_width=True):
                if selected_runs:
                    with st.spinner("Loading survey data..."):
                        self.load_selected_data(selected_runs)
                        st.success(f"Loaded data from {len(selected_runs)} runs")
                else:
                    st.warning("Please select at least one run to load")
                    
    def render_filtering_interface(self):
        """Render the filtering and query interface"""
        st.header("Data Filtering")
        
        if st.session_state.comparison_data is None:
            st.info("Please load data from the Data Selection tab first")
            return
            
        data = st.session_state.comparison_data
        
        # Basic filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Model filter
            available_models = data['model'].unique().tolist()
            selected_models = st.multiselect(
                "Filter by Models:",
                options=available_models,
                default=st.session_state.filter_settings['models'],
                help="Select specific models to include in analysis"
            )
            st.session_state.filter_settings['models'] = selected_models
            
        with col2:
            # Scale filter
            available_scales = data['scale'].unique().tolist()
            selected_scales = st.multiselect(
                "Filter by Scales:",
                options=available_scales,
                default=st.session_state.filter_settings['scales'],
                help="Select specific scales to analyze"
            )
            st.session_state.filter_settings['scales'] = selected_scales
            
        with col3:
            # Prompt filter
            available_prompts = data['prompt'].unique().tolist()
            selected_prompts = st.multiselect(
                "Filter by Prompt Styles:",
                options=available_prompts,
                default=st.session_state.filter_settings['prompts'],
                help="Select specific prompt styles"
            )
            st.session_state.filter_settings['prompts'] = selected_prompts
            
        # Advanced filters
        st.subheader("Advanced Filters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Response type filter
            response_type = st.selectbox(
                "Response Type:",
                options=['all', 'valid', 'refused', 'outliers'],
                index=['all', 'valid', 'refused', 'outliers'].index(
                    st.session_state.filter_settings.get('response_type', 'all')
                ),
                help="Filter by response validity"
            )
            st.session_state.filter_settings['response_type'] = response_type
            
            # Score range filter
            if 'score' in data.columns:
                min_score = float(data['score'].min())
                max_score = float(data['score'].max())
                
                score_range = st.slider(
                    "Score Range:",
                    min_value=min_score,
                    max_value=max_score,
                    value=(min_score, max_score),
                    step=0.1,
                    help="Filter responses by score range"
                )
                st.session_state.filter_settings['score_range'] = score_range
                
        with col2:
            # Date range picker
            if 'timestamp' in data.columns:
                min_date = pd.to_datetime(data['timestamp'].min()).date()
                max_date = pd.to_datetime(data['timestamp'].max()).date()
                
                date_range = st.date_input(
                    "Date Range:",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date,
                    help="Filter by date range"
                )
                
                if len(date_range) == 2:
                    st.session_state.filter_settings['date_range'] = date_range
                    
        # Custom query builder
        with st.expander("Custom Query Builder", expanded=False):
            st.subheader("Build Custom Query")
            
            query_string = st.text_area(
                "Enter pandas query string:",
                placeholder="e.g., score > 3.5 & model == 'gpt-4' & scale == 'RWA'",
                help="Use pandas query syntax for complex filtering"
            )
            
            if st.button("Apply Custom Query"):
                if query_string:
                    try:
                        filtered_data = data.query(query_string)
                        st.success(f"Query applied successfully. {len(filtered_data)} rows match.")
                        st.session_state.filtered_data = filtered_data
                    except Exception as e:
                        st.error(f"Query error: {str(e)}")
                        
        # Apply filters button
        if st.button("Apply All Filters", type="primary", use_container_width=True):
            filtered_data = self.apply_filters(data)
            st.session_state.filtered_data = filtered_data
            st.success(f"Filters applied. {len(filtered_data)} rows in filtered dataset.")
            
    def render_data_views(self):
        """Render different data view options"""
        st.header("Data Views")
        
        if st.session_state.comparison_data is None:
            st.info("Please load data from the Data Selection tab first")
            return
            
        # Use filtered data if available, otherwise use all data
        data = st.session_state.get('filtered_data', st.session_state.comparison_data)
        
        # View selection
        view_type = st.selectbox(
            "Select View Type:",
            options=["Tabular View", "Statistical Summary", "Response Distribution", 
                    "Model Comparison", "Scale Correlations", "Time Series"],
            help="Choose how to view the data"
        )
        
        if view_type == "Tabular View":
            self.render_tabular_view(data)
            
        elif view_type == "Statistical Summary":
            self.render_statistical_summary(data)
            
        elif view_type == "Response Distribution":
            self.render_response_distribution(data)
            
        elif view_type == "Model Comparison":
            self.render_model_comparison(data)
            
        elif view_type == "Scale Correlations":
            self.render_scale_correlations(data)
            
        elif view_type == "Time Series":
            self.render_time_series(data)
            
    def render_tabular_view(self, data: pd.DataFrame):
        """Render sortable tabular view of the data"""
        st.subheader("Tabular Data View")
        
        # Column selection
        all_columns = data.columns.tolist()
        default_columns = ['run_id', 'model', 'scale', 'prompt', 'score', 'timestamp']
        default_columns = [col for col in default_columns if col in all_columns]
        
        selected_columns = st.multiselect(
            "Select columns to display:",
            options=all_columns,
            default=default_columns,
            help="Choose which columns to show in the table"
        )
        
        if selected_columns:
            # Sort options
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                sort_column = st.selectbox(
                    "Sort by:",
                    options=selected_columns,
                    index=0 if selected_columns else None
                )
                
            with col2:
                sort_order = st.radio(
                    "Order:",
                    options=["Ascending", "Descending"],
                    horizontal=True
                )
                
            with col3:
                rows_per_page = st.number_input(
                    "Rows per page:",
                    min_value=10,
                    max_value=100,
                    value=25,
                    step=5
                )
                
            # Apply sorting
            ascending = sort_order == "Ascending"
            sorted_data = data[selected_columns].sort_values(by=sort_column, ascending=ascending)
            
            # Pagination
            total_rows = len(sorted_data)
            total_pages = (total_rows - 1) // rows_per_page + 1
            
            page = st.number_input(
                f"Page (1-{total_pages}):",
                min_value=1,
                max_value=total_pages,
                value=1
            )
            
            start_idx = (page - 1) * rows_per_page
            end_idx = min(start_idx + rows_per_page, total_rows)
            
            # Display table
            st.dataframe(
                sorted_data.iloc[start_idx:end_idx],
                use_container_width=True,
                height=400
            )
            
            st.caption(f"Showing rows {start_idx+1}-{end_idx} of {total_rows}")
            
            # Highlight outliers option
            if st.checkbox("Highlight outliers"):
                self.highlight_outliers(sorted_data.iloc[start_idx:end_idx])
                
    def render_statistical_summary(self, data: pd.DataFrame):
        """Render statistical summary of the data"""
        st.subheader("Statistical Summary")
        
        # Select numeric columns
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_columns:
            st.warning("No numeric columns found in the data")
            return
            
        # Group by options
        groupby_columns = ['model', 'scale', 'prompt', 'run_id']
        available_groupby = [col for col in groupby_columns if col in data.columns]
        
        selected_groupby = st.multiselect(
            "Group statistics by:",
            options=available_groupby,
            default=['model'] if 'model' in available_groupby else [],
            help="Calculate statistics for each group"
        )
        
        # Calculate statistics
        if selected_groupby:
            stats_df = data.groupby(selected_groupby)[numeric_columns].agg([
                'count', 'mean', 'std', 'min', 
                ('25%', lambda x: x.quantile(0.25)),
                ('50%', lambda x: x.quantile(0.50)),
                ('75%', lambda x: x.quantile(0.75)),
                'max'
            ]).round(3)
            
            st.dataframe(stats_df, use_container_width=True)
            
            # Download statistics
            csv = stats_df.to_csv()
            st.download_button(
                "Download Statistics as CSV",
                data=csv,
                file_name=f"statistics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            # Overall statistics
            stats_df = data[numeric_columns].describe().round(3)
            st.dataframe(stats_df, use_container_width=True)
            
    def render_response_distribution(self, data: pd.DataFrame):
        """Render response distribution analysis"""
        st.subheader("Response Distribution Analysis")
        
        if 'score' not in data.columns:
            st.warning("No score column found in the data")
            return
            
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribution type
            dist_type = st.selectbox(
                "Distribution Type:",
                options=["Histogram", "Density Plot", "Box Plot", "Violin Plot"],
                help="Choose visualization type for distribution"
            )
            
        with col2:
            # Group by option
            groupby_options = ['None', 'model', 'scale', 'prompt']
            available_groupby = [opt for opt in groupby_options if opt == 'None' or opt in data.columns]
            
            group_by = st.selectbox(
                "Group by:",
                options=available_groupby,
                help="Compare distributions across groups"
            )
            
        # Create distribution plot
        if dist_type == "Histogram":
            if group_by == 'None':
                fig = px.histogram(
                    data, 
                    x='score',
                    nbins=30,
                    title="Score Distribution",
                    labels={'score': 'Score', 'count': 'Frequency'}
                )
            else:
                fig = px.histogram(
                    data, 
                    x='score',
                    color=group_by,
                    nbins=30,
                    title=f"Score Distribution by {group_by}",
                    labels={'score': 'Score', 'count': 'Frequency'},
                    barmode='overlay',
                    opacity=0.7
                )
                
        elif dist_type == "Density Plot":
            if group_by == 'None':
                fig = px.density_contour(
                    data,
                    x='score',
                    title="Score Density",
                    labels={'score': 'Score'}
                )
            else:
                fig = go.Figure()
                for group in data[group_by].unique():
                    group_data = data[data[group_by] == group]['score']
                    fig.add_trace(go.Violin(
                        x=group_data,
                        name=str(group),
                        orientation='h',
                        side='positive',
                        width=3
                    ))
                fig.update_layout(title=f"Score Density by {group_by}")
                
        elif dist_type == "Box Plot":
            if group_by == 'None':
                fig = px.box(
                    data,
                    y='score',
                    title="Score Distribution",
                    labels={'score': 'Score'}
                )
            else:
                fig = px.box(
                    data,
                    x=group_by,
                    y='score',
                    title=f"Score Distribution by {group_by}",
                    labels={'score': 'Score'}
                )
                
        elif dist_type == "Violin Plot":
            if group_by == 'None':
                fig = px.violin(
                    data,
                    y='score',
                    title="Score Distribution",
                    labels={'score': 'Score'},
                    box=True
                )
            else:
                fig = px.violin(
                    data,
                    x=group_by,
                    y='score',
                    title=f"Score Distribution by {group_by}",
                    labels={'score': 'Score'},
                    box=True
                )
                
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Distribution statistics
        st.subheader("Distribution Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Mean", f"{data['score'].mean():.3f}")
            st.metric("Median", f"{data['score'].median():.3f}")
            
        with col2:
            st.metric("Std Dev", f"{data['score'].std():.3f}")
            st.metric("Variance", f"{data['score'].var():.3f}")
            
        with col3:
            st.metric("Skewness", f"{data['score'].skew():.3f}")
            st.metric("Kurtosis", f"{data['score'].kurtosis():.3f}")
            
        with col4:
            st.metric("Min", f"{data['score'].min():.3f}")
            st.metric("Max", f"{data['score'].max():.3f}")
            
    def render_model_comparison(self, data: pd.DataFrame):
        """Render model comparison charts"""
        st.subheader("Model Comparison")
        
        if 'model' not in data.columns or 'score' not in data.columns:
            st.warning("Required columns (model, score) not found")
            return
            
        # Comparison metric
        metric = st.selectbox(
            "Comparison Metric:",
            options=["Mean Score", "Response Rate", "Score Variance", "Cost per Response"],
            help="Choose metric to compare models"
        )
        
        if metric == "Mean Score":
            # Calculate mean scores by model
            model_scores = data.groupby('model')['score'].agg(['mean', 'std', 'count']).reset_index()
            
            # Create bar chart with error bars
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=model_scores['model'],
                y=model_scores['mean'],
                error_y=dict(
                    type='data',
                    array=model_scores['std'],
                    visible=True
                ),
                text=model_scores['mean'].round(3),
                textposition='outside'
            ))
            
            fig.update_layout(
                title="Mean Score by Model",
                xaxis_title="Model",
                yaxis_title="Mean Score",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistical comparison table
            st.subheader("Statistical Comparison")
            comparison_df = model_scores.rename(columns={
                'mean': 'Mean Score',
                'std': 'Std Dev',
                'count': 'N Responses'
            })
            st.dataframe(comparison_df, use_container_width=True)
            
        elif metric == "Response Rate":
            # Calculate response rates
            if 'is_valid' in data.columns:
                response_rates = data.groupby('model')['is_valid'].agg(['sum', 'count'])
                response_rates['rate'] = (response_rates['sum'] / response_rates['count'] * 100).round(2)
                response_rates = response_rates.reset_index()
                
                fig = px.bar(
                    response_rates,
                    x='model',
                    y='rate',
                    title="Response Rate by Model",
                    labels={'rate': 'Response Rate (%)'},
                    text='rate'
                )
                
                fig.update_traces(texttemplate='%{text}%', textposition='outside')
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
        elif metric == "Score Variance":
            # Calculate variance by model
            model_variance = data.groupby('model')['score'].var().reset_index()
            model_variance.columns = ['model', 'variance']
            
            fig = px.bar(
                model_variance,
                x='model',
                y='variance',
                title="Score Variance by Model",
                labels={'variance': 'Variance'},
                text='variance'
            )
            
            fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
        elif metric == "Cost per Response":
            if 'cost' in data.columns:
                cost_analysis = data.groupby('model')['cost'].agg(['sum', 'count'])
                cost_analysis['cost_per_response'] = (cost_analysis['sum'] / cost_analysis['count']).round(4)
                cost_analysis = cost_analysis.reset_index()
                
                fig = px.bar(
                    cost_analysis,
                    x='model',
                    y='cost_per_response',
                    title="Cost per Response by Model",
                    labels={'cost_per_response': 'Cost ($)'},
                    text='cost_per_response'
                )
                
                fig.update_traces(texttemplate='$%{text}', textposition='outside')
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Cost data not available")
                
    def render_scale_correlations(self, data: pd.DataFrame):
        """Render scale correlation analysis"""
        st.subheader("Scale Correlation Analysis")
        
        if 'scale' not in data.columns or 'score' not in data.columns:
            st.warning("Required columns (scale, score) not found")
            return
            
        # Pivot data for correlation
        pivot_data = data.pivot_table(
            values='score',
            index=data.index,
            columns='scale',
            aggfunc='first'
        )
        
        if pivot_data.shape[1] < 2:
            st.warning("Need at least 2 scales for correlation analysis")
            return
            
        # Calculate correlation matrix
        corr_matrix = pivot_data.corr()
        
        # Create heatmap
        fig = px.imshow(
            corr_matrix,
            labels=dict(x="Scale", y="Scale", color="Correlation"),
            title="Scale Correlation Heatmap",
            color_continuous_scale="RdBu",
            aspect="auto",
            zmin=-1,
            zmax=1
        )
        
        # Add correlation values as text
        fig.update_traces(text=corr_matrix.round(2), texttemplate="%{text}")
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation statistics
        st.subheader("Correlation Statistics")
        
        # Find strongest correlations
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_pairs.append({
                    'Scale 1': corr_matrix.columns[i],
                    'Scale 2': corr_matrix.columns[j],
                    'Correlation': corr_matrix.iloc[i, j]
                })
                
        corr_df = pd.DataFrame(corr_pairs).sort_values('Correlation', key=abs, ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Strongest Positive Correlations:**")
            positive_corr = corr_df[corr_df['Correlation'] > 0].head(5)
            st.dataframe(positive_corr, use_container_width=True, hide_index=True)
            
        with col2:
            st.write("**Strongest Negative Correlations:**")
            negative_corr = corr_df[corr_df['Correlation'] < 0].head(5)
            st.dataframe(negative_corr, use_container_width=True, hide_index=True)
            
    def render_time_series(self, data: pd.DataFrame):
        """Render time series analysis"""
        st.subheader("Time Series Analysis")
        
        if 'timestamp' not in data.columns:
            st.warning("No timestamp column found in the data")
            return
            
        # Convert timestamp to datetime
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        
        # Time aggregation options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            aggregation = st.selectbox(
                "Time Aggregation:",
                options=["Daily", "Weekly", "Monthly"],
                help="Choose time period for aggregation"
            )
            
        with col2:
            metric = st.selectbox(
                "Metric:",
                options=["Mean Score", "Response Count", "Total Cost"],
                help="Choose metric to plot over time"
            )
            
        with col3:
            if 'model' in data.columns:
                group_by = st.selectbox(
                    "Group by:",
                    options=["None", "model", "scale", "prompt"],
                    help="Compare trends across groups"
                )
            else:
                group_by = "None"
                
        # Set date frequency
        if aggregation == "Daily":
            freq = 'D'
        elif aggregation == "Weekly":
            freq = 'W'
        else:
            freq = 'M'
            
        # Prepare time series data
        data_copy = data.copy()
        data_copy.set_index('timestamp', inplace=True)
        
        if metric == "Mean Score" and 'score' in data.columns:
            if group_by == "None":
                ts_data = data_copy['score'].resample(freq).mean()
                
                fig = px.line(
                    x=ts_data.index,
                    y=ts_data.values,
                    title=f"{metric} Over Time ({aggregation})",
                    labels={'x': 'Date', 'y': metric}
                )
            else:
                ts_data = data_copy.groupby(group_by)['score'].resample(freq).mean().reset_index()
                
                fig = px.line(
                    ts_data,
                    x='timestamp',
                    y='score',
                    color=group_by,
                    title=f"{metric} Over Time by {group_by} ({aggregation})",
                    labels={'timestamp': 'Date', 'score': metric}
                )
                
        elif metric == "Response Count":
            if group_by == "None":
                ts_data = data_copy.resample(freq).size()
                
                fig = px.bar(
                    x=ts_data.index,
                    y=ts_data.values,
                    title=f"{metric} Over Time ({aggregation})",
                    labels={'x': 'Date', 'y': metric}
                )
            else:
                ts_data = data_copy.groupby(group_by).resample(freq).size().reset_index(name='count')
                
                fig = px.bar(
                    ts_data,
                    x='timestamp',
                    y='count',
                    color=group_by,
                    title=f"{metric} Over Time by {group_by} ({aggregation})",
                    labels={'timestamp': 'Date', 'count': metric}
                )
                
        elif metric == "Total Cost" and 'cost' in data.columns:
            if group_by == "None":
                ts_data = data_copy['cost'].resample(freq).sum()
                
                fig = px.area(
                    x=ts_data.index,
                    y=ts_data.values,
                    title=f"{metric} Over Time ({aggregation})",
                    labels={'x': 'Date', 'y': metric}
                )
            else:
                ts_data = data_copy.groupby(group_by)['cost'].resample(freq).sum().reset_index()
                
                fig = px.area(
                    ts_data,
                    x='timestamp',
                    y='cost',
                    color=group_by,
                    title=f"{metric} Over Time by {group_by} ({aggregation})",
                    labels={'timestamp': 'Date', 'cost': metric}
                )
        else:
            st.warning(f"Metric '{metric}' not available in the data")
            return
            
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Trend analysis
        if st.checkbox("Show trend analysis"):
            self.render_trend_analysis(ts_data)
            
    def render_visualizations(self):
        """Render advanced visualizations tab"""
        st.header("Advanced Visualizations")
        
        if st.session_state.comparison_data is None:
            st.info("Please load data from the Data Selection tab first")
            return
            
        data = st.session_state.get('filtered_data', st.session_state.comparison_data)
        
        # Visualization type selection
        viz_type = st.selectbox(
            "Select Visualization Type:",
            options=[
                "Interactive Scatter Plot",
                "3D Visualization",
                "Parallel Coordinates",
                "Sunburst Chart",
                "Sankey Diagram",
                "Radar Chart"
            ],
            help="Choose advanced visualization type"
        )
        
        if viz_type == "Interactive Scatter Plot":
            self.render_scatter_plot(data)
            
        elif viz_type == "3D Visualization":
            self.render_3d_visualization(data)
            
        elif viz_type == "Parallel Coordinates":
            self.render_parallel_coordinates(data)
            
        elif viz_type == "Sunburst Chart":
            self.render_sunburst_chart(data)
            
        elif viz_type == "Sankey Diagram":
            self.render_sankey_diagram(data)
            
        elif viz_type == "Radar Chart":
            self.render_radar_chart(data)
            
    def render_scatter_plot(self, data: pd.DataFrame):
        """Render interactive scatter plot"""
        st.subheader("Interactive Scatter Plot")
        
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
        
        if len(numeric_columns) < 2:
            st.warning("Need at least 2 numeric columns for scatter plot")
            return
            
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            x_axis = st.selectbox("X-axis:", options=numeric_columns)
            
        with col2:
            y_axis = st.selectbox("Y-axis:", options=numeric_columns, index=1 if len(numeric_columns) > 1 else 0)
            
        with col3:
            color_by = st.selectbox(
                "Color by:",
                options=["None"] + categorical_columns + numeric_columns,
                help="Color points by category or value"
            )
            
        with col4:
            size_by = st.selectbox(
                "Size by:",
                options=["None"] + numeric_columns,
                help="Size points by numeric value"
            )
            
        # Create scatter plot
        fig_kwargs = {
            'data_frame': data,
            'x': x_axis,
            'y': y_axis,
            'title': f"{y_axis} vs {x_axis}"
        }
        
        if color_by != "None":
            fig_kwargs['color'] = color_by
            
        if size_by != "None":
            fig_kwargs['size'] = size_by
            
        # Add hover data
        hover_data = ['model', 'scale', 'prompt'] if all(col in data.columns for col in ['model', 'scale', 'prompt']) else None
        if hover_data:
            fig_kwargs['hover_data'] = hover_data
            
        fig = px.scatter(**fig_kwargs)
        
        # Add trendline option
        if st.checkbox("Add trendline"):
            fig.add_trace(go.Scatter(
                x=data[x_axis],
                y=data[y_axis],
                mode='lines',
                name='Trendline',
                line=dict(dash='dash')
            ))
            
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
    def render_3d_visualization(self, data: pd.DataFrame):
        """Render 3D visualization"""
        st.subheader("3D Visualization")
        
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_columns) < 3:
            st.warning("Need at least 3 numeric columns for 3D visualization")
            return
            
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            x_axis = st.selectbox("X-axis:", options=numeric_columns)
            
        with col2:
            y_axis = st.selectbox("Y-axis:", options=numeric_columns, index=1 if len(numeric_columns) > 1 else 0)
            
        with col3:
            z_axis = st.selectbox("Z-axis:", options=numeric_columns, index=2 if len(numeric_columns) > 2 else 0)
            
        with col4:
            color_by = st.selectbox(
                "Color by:",
                options=["None"] + data.columns.tolist(),
                help="Color points by category or value"
            )
            
        # Create 3D scatter plot
        fig_kwargs = {
            'data_frame': data,
            'x': x_axis,
            'y': y_axis,
            'z': z_axis,
            'title': f"3D Plot: {x_axis} vs {y_axis} vs {z_axis}"
        }
        
        if color_by != "None":
            fig_kwargs['color'] = color_by
            
        fig = px.scatter_3d(**fig_kwargs)
        fig.update_layout(height=700)
        st.plotly_chart(fig, use_container_width=True)
        
    def render_parallel_coordinates(self, data: pd.DataFrame):
        """Render parallel coordinates plot"""
        st.subheader("Parallel Coordinates Plot")
        
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_columns) < 2:
            st.warning("Need at least 2 numeric columns for parallel coordinates")
            return
            
        # Select dimensions
        selected_dims = st.multiselect(
            "Select dimensions:",
            options=numeric_columns,
            default=numeric_columns[:5] if len(numeric_columns) >= 5 else numeric_columns,
            help="Choose dimensions for parallel coordinates plot"
        )
        
        if len(selected_dims) < 2:
            st.warning("Please select at least 2 dimensions")
            return
            
        # Color by option
        color_column = st.selectbox(
            "Color by:",
            options=["None"] + data.columns.tolist(),
            help="Color lines by category or value"
        )
        
        # Prepare dimensions
        dimensions = []
        for col in selected_dims:
            dimensions.append(
                dict(
                    label=col,
                    values=data[col],
                    range=[data[col].min(), data[col].max()]
                )
            )
            
        # Create parallel coordinates plot
        if color_column != "None" and color_column in numeric_columns:
            line_color = data[color_column]
            colorscale = 'Viridis'
        else:
            line_color = None
            colorscale = None
            
        fig = go.Figure(data=go.Parcoords(
            dimensions=dimensions,
            line=dict(
                color=line_color,
                colorscale=colorscale,
                showscale=True if line_color is not None else False
            )
        ))
        
        fig.update_layout(
            title="Parallel Coordinates Plot",
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    def render_sunburst_chart(self, data: pd.DataFrame):
        """Render sunburst chart for hierarchical data"""
        st.subheader("Sunburst Chart")
        
        categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
        
        if len(categorical_columns) < 2:
            st.warning("Need at least 2 categorical columns for sunburst chart")
            return
            
        # Select hierarchy
        hierarchy = st.multiselect(
            "Select hierarchy (from outer to inner):",
            options=categorical_columns,
            default=categorical_columns[:3] if len(categorical_columns) >= 3 else categorical_columns,
            help="Define the hierarchy for the sunburst chart"
        )
        
        if len(hierarchy) < 2:
            st.warning("Please select at least 2 levels for hierarchy")
            return
            
        # Value column
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_columns:
            value_column = st.selectbox(
                "Value column (for sizing):",
                options=["Count"] + numeric_columns,
                help="Choose value for sizing segments"
            )
        else:
            value_column = "Count"
            
        # Prepare data for sunburst
        if value_column == "Count":
            sunburst_data = data.groupby(hierarchy).size().reset_index(name='count')
            values = 'count'
        else:
            sunburst_data = data.groupby(hierarchy)[value_column].sum().reset_index()
            values = value_column
            
        # Create sunburst chart
        fig = px.sunburst(
            sunburst_data,
            path=hierarchy,
            values=values,
            title="Hierarchical Data Visualization"
        )
        
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
    def render_sankey_diagram(self, data: pd.DataFrame):
        """Render Sankey diagram for flow visualization"""
        st.subheader("Sankey Diagram")
        
        categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
        
        if len(categorical_columns) < 2:
            st.warning("Need at least 2 categorical columns for Sankey diagram")
            return
            
        col1, col2 = st.columns(2)
        
        with col1:
            source_column = st.selectbox(
                "Source:",
                options=categorical_columns,
                help="Select source nodes"
            )
            
        with col2:
            target_column = st.selectbox(
                "Target:",
                options=[col for col in categorical_columns if col != source_column],
                help="Select target nodes"
            )
            
        # Prepare Sankey data
        flow_data = data.groupby([source_column, target_column]).size().reset_index(name='count')
        
        # Create node labels
        source_labels = flow_data[source_column].unique()
        target_labels = flow_data[target_column].unique()
        all_labels = list(source_labels) + list(target_labels)
        
        # Create node indices
        label_to_idx = {label: idx for idx, label in enumerate(all_labels)}
        
        # Create source and target indices
        source_idx = [label_to_idx[label] for label in flow_data[source_column]]
        target_idx = [label_to_idx[label] for label in flow_data[target_column]]
        
        # Create Sankey diagram
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=all_labels
            ),
            link=dict(
                source=source_idx,
                target=target_idx,
                value=flow_data['count']
            )
        )])
        
        fig.update_layout(
            title=f"Flow from {source_column} to {target_column}",
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    def render_radar_chart(self, data: pd.DataFrame):
        """Render radar chart for multi-dimensional comparison"""
        st.subheader("Radar Chart")
        
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_columns) < 3:
            st.warning("Need at least 3 numeric columns for radar chart")
            return
            
        # Select dimensions
        selected_dims = st.multiselect(
            "Select dimensions:",
            options=numeric_columns,
            default=numeric_columns[:6] if len(numeric_columns) >= 6 else numeric_columns,
            help="Choose dimensions for radar chart"
        )
        
        if len(selected_dims) < 3:
            st.warning("Please select at least 3 dimensions")
            return
            
        # Group by option
        if 'model' in data.columns:
            group_by = st.selectbox(
                "Compare by:",
                options=['model', 'scale', 'prompt'],
                help="Compare different groups on radar chart"
            )
        else:
            group_by = None
            
        # Create radar chart
        fig = go.Figure()
        
        if group_by and group_by in data.columns:
            for group in data[group_by].unique():
                group_data = data[data[group_by] == group]
                values = [group_data[dim].mean() for dim in selected_dims]
                
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=selected_dims,
                    fill='toself',
                    name=str(group)
                ))
        else:
            values = [data[dim].mean() for dim in selected_dims]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=selected_dims,
                fill='toself',
                name='Overall'
            ))
            
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(data[selected_dims].max())]
                )
            ),
            showlegend=True,
            title="Multi-dimensional Comparison",
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    def render_statistical_analysis(self):
        """Render statistical analysis tab"""
        st.header("Statistical Analysis")
        
        if st.session_state.comparison_data is None:
            st.info("Please load data from the Data Selection tab first")
            return
            
        data = st.session_state.get('filtered_data', st.session_state.comparison_data)
        
        # Analysis type selection
        analysis_type = st.selectbox(
            "Select Analysis Type:",
            options=[
                "T-Test",
                "ANOVA",
                "Correlation Analysis",
                "Regression Analysis",
                "Outlier Detection",
                "Consistency Metrics"
            ],
            help="Choose statistical analysis to perform"
        )
        
        if analysis_type == "T-Test":
            self.render_t_test(data)
            
        elif analysis_type == "ANOVA":
            self.render_anova(data)
            
        elif analysis_type == "Correlation Analysis":
            self.render_correlation_analysis(data)
            
        elif analysis_type == "Regression Analysis":
            self.render_regression_analysis(data)
            
        elif analysis_type == "Outlier Detection":
            self.render_outlier_detection(data)
            
        elif analysis_type == "Consistency Metrics":
            self.render_consistency_metrics(data)
            
    def render_t_test(self, data: pd.DataFrame):
        """Render t-test analysis"""
        st.subheader("T-Test Analysis")
        
        if 'score' not in data.columns:
            st.warning("Score column not found in data")
            return
            
        # Select grouping variable
        categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
        
        group_column = st.selectbox(
            "Select grouping variable:",
            options=categorical_columns,
            help="Variable to define the two groups for comparison"
        )
        
        # Get unique groups
        groups = data[group_column].unique()
        
        if len(groups) < 2:
            st.warning("Need at least 2 groups for t-test")
            return
            
        col1, col2 = st.columns(2)
        
        with col1:
            group1 = st.selectbox("Group 1:", options=groups)
            
        with col2:
            group2 = st.selectbox("Group 2:", options=[g for g in groups if g != group1])
            
        # Perform t-test
        group1_data = data[data[group_column] == group1]['score']
        group2_data = data[data[group_column] == group2]['score']
        
        t_stat, p_value = stats.ttest_ind(group1_data, group2_data)
        
        # Display results
        st.subheader("Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("T-statistic", f"{t_stat:.4f}")
            
        with col2:
            st.metric("P-value", f"{p_value:.4f}")
            
        with col3:
            significance = "Significant" if p_value < 0.05 else "Not Significant"
            st.metric("Result (α=0.05)", significance)
            
        # Group statistics
        st.subheader("Group Statistics")
        
        stats_df = pd.DataFrame({
            'Group': [group1, group2],
            'N': [len(group1_data), len(group2_data)],
            'Mean': [group1_data.mean(), group2_data.mean()],
            'Std Dev': [group1_data.std(), group2_data.std()],
            'Min': [group1_data.min(), group2_data.min()],
            'Max': [group1_data.max(), group2_data.max()]
        })
        
        st.dataframe(stats_df, use_container_width=True)
        
        # Visualization
        fig = go.Figure()
        
        fig.add_trace(go.Box(y=group1_data, name=str(group1)))
        fig.add_trace(go.Box(y=group2_data, name=str(group2)))
        
        fig.update_layout(
            title="Group Comparison",
            yaxis_title="Score",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    def render_anova(self, data: pd.DataFrame):
        """Render ANOVA analysis"""
        st.subheader("ANOVA Analysis")
        
        if 'score' not in data.columns:
            st.warning("Score column not found in data")
            return
            
        # Select grouping variable
        categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
        
        group_column = st.selectbox(
            "Select grouping variable:",
            options=categorical_columns,
            help="Variable to define groups for ANOVA"
        )
        
        # Prepare data for ANOVA
        groups = []
        group_names = []
        
        for group_name in data[group_column].unique():
            group_data = data[data[group_column] == group_name]['score'].dropna()
            if len(group_data) > 0:
                groups.append(group_data)
                group_names.append(group_name)
                
        if len(groups) < 2:
            st.warning("Need at least 2 groups with data for ANOVA")
            return
            
        # Perform ANOVA
        f_stat, p_value = stats.f_oneway(*groups)
        
        # Display results
        st.subheader("Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("F-statistic", f"{f_stat:.4f}")
            
        with col2:
            st.metric("P-value", f"{p_value:.4f}")
            
        with col3:
            significance = "Significant" if p_value < 0.05 else "Not Significant"
            st.metric("Result (α=0.05)", significance)
            
        # Group statistics
        st.subheader("Group Statistics")
        
        stats_data = []
        for name, group in zip(group_names, groups):
            stats_data.append({
                'Group': name,
                'N': len(group),
                'Mean': group.mean(),
                'Std Dev': group.std(),
                'Min': group.min(),
                'Max': group.max()
            })
            
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, use_container_width=True)
        
        # Post-hoc analysis if significant
        if p_value < 0.05 and st.checkbox("Perform post-hoc analysis (Tukey HSD)"):
            st.subheader("Post-hoc Analysis")
            st.info("Pairwise comparisons between groups")
            
            # This would require additional implementation
            st.write("Post-hoc analysis would show pairwise comparisons here")
            
    def render_correlation_analysis(self, data: pd.DataFrame):
        """Render correlation analysis"""
        st.subheader("Correlation Analysis")
        
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_columns) < 2:
            st.warning("Need at least 2 numeric columns for correlation analysis")
            return
            
        # Select variables
        col1, col2 = st.columns(2)
        
        with col1:
            var1 = st.selectbox("Variable 1:", options=numeric_columns)
            
        with col2:
            var2 = st.selectbox(
                "Variable 2:",
                options=[col for col in numeric_columns if col != var1]
            )
            
        # Calculate correlation
        correlation = data[var1].corr(data[var2])
        
        # Perform significance test
        n = len(data)
        t_stat = correlation * np.sqrt(n - 2) / np.sqrt(1 - correlation**2)
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
        
        # Display results
        st.subheader("Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Correlation (r)", f"{correlation:.4f}")
            
        with col2:
            st.metric("P-value", f"{p_value:.4f}")
            
        with col3:
            if abs(correlation) < 0.3:
                strength = "Weak"
            elif abs(correlation) < 0.7:
                strength = "Moderate"
            else:
                strength = "Strong"
            st.metric("Strength", strength)
            
        # Scatter plot with regression line
        fig = px.scatter(
            data,
            x=var1,
            y=var2,
            title=f"Correlation: {var1} vs {var2}",
            trendline="ols"
        )
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
    def render_regression_analysis(self, data: pd.DataFrame):
        """Render regression analysis"""
        st.subheader("Regression Analysis")
        
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_columns) < 2:
            st.warning("Need at least 2 numeric columns for regression analysis")
            return
            
        # Select variables
        dependent_var = st.selectbox(
            "Dependent Variable (Y):",
            options=numeric_columns,
            help="Variable to predict"
        )
        
        independent_vars = st.multiselect(
            "Independent Variables (X):",
            options=[col for col in numeric_columns if col != dependent_var],
            help="Variables to use as predictors"
        )
        
        if not independent_vars:
            st.warning("Please select at least one independent variable")
            return
            
        # Prepare data
        X = data[independent_vars].dropna()
        y = data.loc[X.index, dependent_var]
        
        # Simple linear regression for visualization
        if len(independent_vars) == 1:
            from sklearn.linear_model import LinearRegression
            
            model = LinearRegression()
            X_array = X.values.reshape(-1, 1)
            model.fit(X_array, y)
            
            # Get predictions
            y_pred = model.predict(X_array)
            
            # Calculate R-squared
            from sklearn.metrics import r2_score
            r2 = r2_score(y, y_pred)
            
            # Display results
            st.subheader("Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("R-squared", f"{r2:.4f}")
                
            with col2:
                st.metric("Coefficient", f"{model.coef_[0]:.4f}")
                
            with col3:
                st.metric("Intercept", f"{model.intercept_:.4f}")
                
            # Visualization
            fig = go.Figure()
            
            # Scatter plot
            fig.add_trace(go.Scatter(
                x=X.iloc[:, 0],
                y=y,
                mode='markers',
                name='Data',
                marker=dict(size=8)
            ))
            
            # Regression line
            fig.add_trace(go.Scatter(
                x=X.iloc[:, 0],
                y=y_pred,
                mode='lines',
                name='Regression Line',
                line=dict(color='red', width=2)
            ))
            
            fig.update_layout(
                title=f"Regression: {dependent_var} vs {independent_vars[0]}",
                xaxis_title=independent_vars[0],
                yaxis_title=dependent_var,
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.info("Multiple regression analysis - showing summary statistics")
            
            from sklearn.linear_model import LinearRegression
            
            model = LinearRegression()
            model.fit(X, y)
            
            # Get predictions
            y_pred = model.predict(X)
            
            # Calculate R-squared
            from sklearn.metrics import r2_score
            r2 = r2_score(y, y_pred)
            
            # Display results
            st.subheader("Results")
            
            st.metric("R-squared", f"{r2:.4f}")
            
            # Coefficients table
            coef_df = pd.DataFrame({
                'Variable': independent_vars,
                'Coefficient': model.coef_
            })
            
            st.dataframe(coef_df, use_container_width=True)
            st.metric("Intercept", f"{model.intercept_:.4f}")
            
    def render_outlier_detection(self, data: pd.DataFrame):
        """Render outlier detection analysis"""
        st.subheader("Outlier Detection")
        
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_columns:
            st.warning("No numeric columns found for outlier detection")
            return
            
        # Select column for analysis
        column = st.selectbox(
            "Select column for outlier detection:",
            options=numeric_columns,
            help="Choose column to analyze for outliers"
        )
        
        # Detection method
        method = st.selectbox(
            "Detection Method:",
            options=["IQR (Interquartile Range)", "Z-Score", "Isolation Forest"],
            help="Choose method for detecting outliers"
        )
        
        outliers = []
        
        if method == "IQR (Interquartile Range)":
            Q1 = data[column].quantile(0.25)
            Q3 = data[column].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
            
            # Display statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Q1", f"{Q1:.3f}")
                
            with col2:
                st.metric("Q3", f"{Q3:.3f}")
                
            with col3:
                st.metric("Lower Bound", f"{lower_bound:.3f}")
                
            with col4:
                st.metric("Upper Bound", f"{upper_bound:.3f}")
                
        elif method == "Z-Score":
            threshold = st.slider(
                "Z-Score Threshold:",
                min_value=2.0,
                max_value=4.0,
                value=3.0,
                step=0.1,
                help="Data points with |z-score| > threshold are outliers"
            )
            
            z_scores = np.abs(stats.zscore(data[column].dropna()))
            outlier_indices = np.where(z_scores > threshold)[0]
            outliers = data.iloc[outlier_indices]
            
        elif method == "Isolation Forest":
            from sklearn.ensemble import IsolationForest
            
            contamination = st.slider(
                "Contamination:",
                min_value=0.01,
                max_value=0.2,
                value=0.1,
                step=0.01,
                help="Expected proportion of outliers"
            )
            
            clf = IsolationForest(contamination=contamination, random_state=42)
            outlier_labels = clf.fit_predict(data[[column]].dropna())
            outliers = data[outlier_labels == -1]
            
        # Display results
        st.subheader("Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Data Points", len(data))
            
        with col2:
            st.metric("Outliers Detected", len(outliers))
            
        # Visualization
        fig = go.Figure()
        
        # All data
        fig.add_trace(go.Box(
            y=data[column],
            name="All Data",
            boxpoints='all',
            jitter=0.3,
            pointpos=-1.8
        ))
        
        # Highlight outliers
        if len(outliers) > 0:
            fig.add_trace(go.Scatter(
                y=outliers[column],
                mode='markers',
                name='Outliers',
                marker=dict(color='red', size=10)
            ))
            
        fig.update_layout(
            title=f"Outlier Detection: {column}",
            yaxis_title=column,
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show outlier data
        if len(outliers) > 0 and st.checkbox("Show outlier data"):
            st.subheader("Outlier Data Points")
            st.dataframe(outliers, use_container_width=True)
            
    def render_consistency_metrics(self, data: pd.DataFrame):
        """Render response consistency metrics"""
        st.subheader("Response Consistency Analysis")
        
        # Check for required columns
        if 'model' not in data.columns or 'score' not in data.columns:
            st.warning("Required columns (model, score) not found")
            return
            
        # Calculate consistency metrics by model
        consistency_metrics = []
        
        for model in data['model'].unique():
            model_data = data[data['model'] == model]['score']
            
            consistency_metrics.append({
                'Model': model,
                'N Responses': len(model_data),
                'Mean': model_data.mean(),
                'Std Dev': model_data.std(),
                'CV (%)': (model_data.std() / model_data.mean() * 100) if model_data.mean() != 0 else 0,
                'Range': model_data.max() - model_data.min(),
                'IQR': model_data.quantile(0.75) - model_data.quantile(0.25)
            })
            
        metrics_df = pd.DataFrame(consistency_metrics)
        
        # Display metrics
        st.dataframe(metrics_df, use_container_width=True)
        
        # Visualization
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Coefficient of Variation", "Interquartile Range")
        )
        
        # CV plot
        fig.add_trace(
            go.Bar(x=metrics_df['Model'], y=metrics_df['CV (%)']),
            row=1, col=1
        )
        
        # IQR plot
        fig.add_trace(
            go.Bar(x=metrics_df['Model'], y=metrics_df['IQR']),
            row=1, col=2
        )
        
        fig.update_xaxes(title_text="Model", row=1, col=1)
        fig.update_xaxes(title_text="Model", row=1, col=2)
        fig.update_yaxes(title_text="CV (%)", row=1, col=1)
        fig.update_yaxes(title_text="IQR", row=1, col=2)
        
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Interpretation
        st.subheader("Interpretation")
        
        most_consistent = metrics_df.loc[metrics_df['CV (%)'].idxmin(), 'Model']
        least_consistent = metrics_df.loc[metrics_df['CV (%)'].idxmax(), 'Model']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.success(f"Most Consistent: {most_consistent}")
            st.caption("Lowest coefficient of variation")
            
        with col2:
            st.warning(f"Least Consistent: {least_consistent}")
            st.caption("Highest coefficient of variation")
            
    def render_export_options(self):
        """Render export and sharing options"""
        st.header("Export & Share")
        
        if st.session_state.comparison_data is None:
            st.info("Please load data from the Data Selection tab first")
            return
            
        data = st.session_state.get('filtered_data', st.session_state.comparison_data)
        
        st.subheader("Export Data")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # CSV export
            csv = data.to_csv(index=False)
            st.download_button(
                label="Download as CSV",
                data=csv,
                file_name=f"survey_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
            
        with col2:
            # JSON export
            json_str = data.to_json(orient='records', indent=2)
            st.download_button(
                label="Download as JSON",
                data=json_str,
                file_name=f"survey_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
            
        with col3:
            # Excel export
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                # Main data sheet
                data.to_excel(writer, sheet_name='Data', index=False)
                
                # Statistics sheet
                stats_df = data.describe()
                stats_df.to_excel(writer, sheet_name='Statistics')
                
                # Model comparison sheet
                if 'model' in data.columns and 'score' in data.columns:
                    model_stats = data.groupby('model')['score'].agg(['mean', 'std', 'count'])
                    model_stats.to_excel(writer, sheet_name='Model Comparison')
                    
            excel_data = output.getvalue()
            
            st.download_button(
                label="Download as Excel",
                data=excel_data,
                file_name=f"survey_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
            
        with col4:
            # Statistical report
            if st.button("Generate Report", use_container_width=True):
                report = self.generate_statistical_report(data)
                st.download_button(
                    label="Download Report",
                    data=report,
                    file_name=f"statistical_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
                
        # Share options
        st.subheader("Share Analysis")
        
        # Generate shareable link (placeholder)
        if st.button("Generate Share Link", use_container_width=True):
            # This would save the current view configuration and generate a link
            share_id = self.generate_share_link(data)
            st.success(f"Share link generated: {share_id}")
            st.code(f"http://yourapp.com/shared/{share_id}")
            
        # Export configuration
        with st.expander("Export Configuration", expanded=False):
            st.json({
                'filters': st.session_state.filter_settings,
                'selected_runs': st.session_state.selected_runs,
                'data_shape': data.shape,
                'timestamp': datetime.now().isoformat()
            })
            
    # Helper methods
    
    def load_available_runs(self) -> Dict[str, Any]:
        """Load metadata for all available survey runs"""
        runs = {}
        
        if not self.data_dir.exists():
            return runs
            
        for run_file in self.data_dir.glob("*.json"):
            try:
                with open(run_file, 'r') as f:
                    run_data = json.load(f)
                    run_id = run_file.stem
                    runs[run_id] = {
                        'id': run_id,
                        'timestamp': run_data.get('timestamp', ''),
                        'config': run_data.get('config', {}),
                        'summary': run_data.get('summary', {}),
                        'file_path': str(run_file)
                    }
            except Exception as e:
                st.warning(f"Error loading run {run_file.name}: {str(e)}")
                
        return runs
        
    def create_runs_dataframe(self, runs: Dict[str, Any]) -> pd.DataFrame:
        """Create DataFrame from runs metadata"""
        data = []
        
        for run_id, run_info in runs.items():
            config = run_info.get('config', {})
            summary = run_info.get('summary', {})
            
            data.append({
                'Run ID': run_id,
                'Date': pd.to_datetime(run_info.get('timestamp', '')).strftime('%Y-%m-%d %H:%M') if run_info.get('timestamp') else 'Unknown',
                'Models': ', '.join(config.get('models', [])),
                'Scales': ', '.join(config.get('scales', [])),
                'Prompts': ', '.join(config.get('prompts', [])),
                'Total Responses': summary.get('total_responses', 0),
                'Valid Responses': summary.get('valid_responses', 0),
                'Success Rate': f"{summary.get('success_rate', 0):.1f}%"
            })
            
        df = pd.DataFrame(data)
        df.set_index('Run ID', inplace=True)
        return df
        
    def search_runs(self, runs: Dict[str, Any], search_term: str) -> List[str]:
        """Search runs by ID or tags"""
        matches = []
        search_lower = search_term.lower()
        
        for run_id, run_info in runs.items():
            # Search in run ID
            if search_lower in run_id.lower():
                matches.append(run_id)
                continue
                
            # Search in configuration
            config = run_info.get('config', {})
            config_str = json.dumps(config).lower()
            if search_lower in config_str:
                matches.append(run_id)
                
        return matches
        
    def apply_date_filter(self, days: int):
        """Apply date filter for recent runs"""
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days)
        st.session_state.filter_settings['date_range'] = (start_date, end_date)
        
    def apply_current_month_filter(self):
        """Apply filter for current month"""
        now = datetime.now()
        start_date = datetime(now.year, now.month, 1).date()
        end_date = now.date()
        st.session_state.filter_settings['date_range'] = (start_date, end_date)
        
    def load_selected_data(self, run_ids: List[str]):
        """Load data from selected runs"""
        all_data = []
        
        for run_id in run_ids:
            run_file = self.data_dir / f"{run_id}.json"
            
            if run_file.exists():
                try:
                    with open(run_file, 'r') as f:
                        run_data = json.load(f)
                        
                    # Extract responses
                    responses = run_data.get('responses', [])
                    
                    for response in responses:
                        # Add run metadata
                        response['run_id'] = run_id
                        response['timestamp'] = run_data.get('timestamp', '')
                        all_data.append(response)
                        
                except Exception as e:
                    st.error(f"Error loading run {run_id}: {str(e)}")
                    
        if all_data:
            df = pd.DataFrame(all_data)
            st.session_state.comparison_data = df
        else:
            st.warning("No data found in selected runs")
            
    def apply_filters(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply all filters to the data"""
        filtered = data.copy()
        
        # Model filter
        if st.session_state.filter_settings['models']:
            filtered = filtered[filtered['model'].isin(st.session_state.filter_settings['models'])]
            
        # Scale filter
        if st.session_state.filter_settings['scales']:
            filtered = filtered[filtered['scale'].isin(st.session_state.filter_settings['scales'])]
            
        # Prompt filter
        if st.session_state.filter_settings['prompts']:
            filtered = filtered[filtered['prompt'].isin(st.session_state.filter_settings['prompts'])]
            
        # Response type filter
        response_type = st.session_state.filter_settings['response_type']
        if response_type == 'valid' and 'is_valid' in filtered.columns:
            filtered = filtered[filtered['is_valid'] == True]
        elif response_type == 'refused' and 'is_valid' in filtered.columns:
            filtered = filtered[filtered['is_valid'] == False]
        elif response_type == 'outliers' and 'score' in filtered.columns:
            # Detect outliers using IQR
            Q1 = filtered['score'].quantile(0.25)
            Q3 = filtered['score'].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            filtered = filtered[(filtered['score'] < lower) | (filtered['score'] > upper)]
            
        # Score range filter
        if st.session_state.filter_settings['score_range'] and 'score' in filtered.columns:
            min_score, max_score = st.session_state.filter_settings['score_range']
            filtered = filtered[(filtered['score'] >= min_score) & (filtered['score'] <= max_score)]
            
        # Date range filter
        if st.session_state.filter_settings['date_range'] and 'timestamp' in filtered.columns:
            start_date, end_date = st.session_state.filter_settings['date_range']
            filtered['date'] = pd.to_datetime(filtered['timestamp']).dt.date
            filtered = filtered[(filtered['date'] >= start_date) & (filtered['date'] <= end_date)]
            
        return filtered
        
    def highlight_outliers(self, data: pd.DataFrame):
        """Highlight outliers in the data"""
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            
            outliers = data[(data[col] < lower) | (data[col] > upper)]
            
            if not outliers.empty:
                st.warning(f"Found {len(outliers)} outliers in {col}")
                
    def render_trend_analysis(self, ts_data):
        """Render trend analysis for time series data"""
        st.subheader("Trend Analysis")
        
        # Calculate moving average
        if isinstance(ts_data, pd.Series):
            ma_7 = ts_data.rolling(window=7).mean()
            ma_30 = ts_data.rolling(window=30).mean()
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=ts_data.index,
                y=ts_data.values,
                mode='lines',
                name='Actual'
            ))
            
            fig.add_trace(go.Scatter(
                x=ma_7.index,
                y=ma_7.values,
                mode='lines',
                name='7-day MA',
                line=dict(dash='dash')
            ))
            
            fig.add_trace(go.Scatter(
                x=ma_30.index,
                y=ma_30.values,
                mode='lines',
                name='30-day MA',
                line=dict(dash='dot')
            ))
            
            fig.update_layout(
                title="Trend Analysis with Moving Averages",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
    def generate_statistical_report(self, data: pd.DataFrame) -> str:
        """Generate a comprehensive statistical report"""
        report = []
        report.append("=" * 50)
        report.append("STATISTICAL ANALYSIS REPORT")
        report.append("=" * 50)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total Records: {len(data)}")
        report.append("")
        
        # Overall statistics
        report.append("OVERALL STATISTICS")
        report.append("-" * 30)
        
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            report.append(f"\n{col}:")
            report.append(f"  Mean: {data[col].mean():.4f}")
            report.append(f"  Std Dev: {data[col].std():.4f}")
            report.append(f"  Min: {data[col].min():.4f}")
            report.append(f"  Max: {data[col].max():.4f}")
            
        # Model comparison
        if 'model' in data.columns and 'score' in data.columns:
            report.append("\nMODEL COMPARISON")
            report.append("-" * 30)
            
            model_stats = data.groupby('model')['score'].agg(['mean', 'std', 'count'])
            for model in model_stats.index:
                report.append(f"\n{model}:")
                report.append(f"  Mean Score: {model_stats.loc[model, 'mean']:.4f}")
                report.append(f"  Std Dev: {model_stats.loc[model, 'std']:.4f}")
                report.append(f"  N: {model_stats.loc[model, 'count']}")
                
        return "\n".join(report)
        
    def generate_share_link(self, data: pd.DataFrame) -> str:
        """Generate a shareable link for the current analysis"""
        # This is a placeholder - in production, you'd save the configuration
        # to a database and return a unique ID
        import hashlib
        
        config_str = json.dumps({
            'filters': st.session_state.filter_settings,
            'selected_runs': st.session_state.selected_runs,
            'timestamp': datetime.now().isoformat()
        })
        
        share_id = hashlib.md5(config_str.encode()).hexdigest()[:8]
        return share_id


# Example usage
if __name__ == "__main__":
    explorer = DataExplorer()
    explorer.render()