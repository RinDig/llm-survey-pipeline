"""Demo application showcasing the Survey Executor component.

This demo provides a standalone Streamlit app that demonstrates the full
capabilities of the survey execution interface with real-time progress tracking.
"""

import streamlit as st
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from frontend.components import (
    render_integrated_executor,
    get_execution_results,
    clear_execution_state
)


def main():
    """Main demo application."""
    st.set_page_config(
        page_title="LLM Survey Executor Demo",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
        <style>
        .stProgress > div > div > div > div {
            background-color: #00cc88;
        }
        .success-metric {
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            border-radius: 4px;
            padding: 10px;
        }
        .error-metric {
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
            border-radius: 4px;
            padding: 10px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.title("🚀 LLM Survey Executor")
    st.markdown("""
        **Real-time Survey Execution & Monitoring System**
        
        This advanced interface provides comprehensive control over LLM survey execution with:
        - 📊 Real-time progress tracking
        - 💰 Live cost accumulation
        - 🔄 Automatic retry handling
        - ⏸️ Pause/Resume functionality
        - 📈 Per-model progress monitoring
        - 💾 Partial results saving
    """)
    
    # Sidebar navigation
    with st.sidebar:
        st.header("🧭 Navigation")
        
        page = st.radio(
            "Select View",
            ["🚀 Survey Executor", "📊 Results Viewer", "📚 Documentation", "🔧 Settings"],
            index=0
        )
        
        st.markdown("---")
        
        # Quick stats if execution is running
        if 'execution_state' in st.session_state:
            state = st.session_state.execution_state
            st.header("📈 Quick Stats")
            
            if state['status'] == 'running':
                st.success(f"Status: {state['status'].upper()}")
                if state['total_tasks'] > 0:
                    progress = (state['completed_tasks'] / state['total_tasks']) * 100
                    st.metric("Progress", f"{progress:.1f}%")
                    st.metric("Tasks", f"{state['completed_tasks']}/{state['total_tasks']}")
            elif state['status'] == 'completed':
                st.info("Status: COMPLETED")
                st.metric("Total Cost", f"${state.get('cost_accumulator', 0):.2f}")
            elif state['status'] == 'paused':
                st.warning("Status: PAUSED")
            elif state['status'] == 'error':
                st.error("Status: ERROR")
        
        st.markdown("---")
        
        # Quick actions
        st.header("⚡ Quick Actions")
        
        if st.button("🗑️ Clear All Data", use_container_width=True):
            clear_execution_state()
            st.success("All data cleared!")
            st.rerun()
        
        if st.button("🔄 Refresh Page", use_container_width=True):
            st.rerun()
    
    # Main content based on selected page
    if page == "🚀 Survey Executor":
        render_integrated_executor()
    
    elif page == "📊 Results Viewer":
        st.header("📊 Survey Results Viewer")
        
        results_data = get_execution_results()
        
        if results_data:
            # Success banner
            st.success("✅ Survey results available for analysis")
            
            # Metadata summary
            metadata = results_data['metadata']
            
            st.subheader("📈 Execution Summary")
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric(
                    "Total Tasks",
                    metadata['total_tasks'],
                    help="Total number of survey tasks executed"
                )
            
            with col2:
                completed = metadata['completed_tasks']
                total = metadata['total_tasks']
                success_rate = (completed / total * 100) if total > 0 else 0
                st.metric(
                    "Success Rate",
                    f"{success_rate:.1f}%",
                    help="Percentage of successfully completed tasks"
                )
            
            with col3:
                st.metric(
                    "Failed Tasks",
                    metadata['failed_tasks'],
                    help="Number of tasks that encountered errors"
                )
            
            with col4:
                st.metric(
                    "Total Cost",
                    f"${metadata['total_cost']:.3f}",
                    help="Estimated total API cost"
                )
            
            with col5:
                if metadata['start_time'] and metadata['end_time']:
                    duration = metadata['end_time'] - metadata['start_time']
                    minutes = duration.total_seconds() / 60
                    st.metric(
                        "Duration",
                        f"{minutes:.1f}m",
                        help="Total execution time"
                    )
            
            # Token usage breakdown
            if metadata.get('token_usage'):
                st.subheader("🎫 Token Usage by Model")
                
                cols = st.columns(len(metadata['token_usage']))
                for idx, (model, tokens) in enumerate(metadata['token_usage'].items()):
                    with cols[idx]:
                        st.metric(model, f"{tokens:,} tokens")
            
            # Results data
            st.subheader("📋 Survey Response Data")
            
            results_df = results_data['results']
            
            # Data filters
            with st.expander("🔍 Filter Options", expanded=False):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    model_filter = st.multiselect(
                        "Filter by Model",
                        results_df['model_name'].unique() if 'model_name' in results_df else []
                    )
                
                with col2:
                    scale_filter = st.multiselect(
                        "Filter by Scale",
                        results_df['scale_name'].unique() if 'scale_name' in results_df else []
                    )
                
                with col3:
                    prompt_filter = st.multiselect(
                        "Filter by Prompt",
                        results_df['prompt_style'].unique() if 'prompt_style' in results_df else []
                    )
                
                # Apply filters
                filtered_df = results_df.copy()
                if model_filter:
                    filtered_df = filtered_df[filtered_df['model_name'].isin(model_filter)]
                if scale_filter:
                    filtered_df = filtered_df[filtered_df['scale_name'].isin(scale_filter)]
                if prompt_filter:
                    filtered_df = filtered_df[filtered_df['prompt_style'].isin(prompt_filter)]
            else:
                filtered_df = results_df
            
            # Display options
            col1, col2 = st.columns([3, 1])
            with col2:
                display_rows = st.number_input(
                    "Rows to display",
                    min_value=10,
                    max_value=len(filtered_df),
                    value=min(50, len(filtered_df)),
                    step=10
                )
            
            # Show dataframe
            st.dataframe(
                filtered_df.head(display_rows),
                use_container_width=True,
                height=400
            )
            
            # Download options
            st.subheader("💾 Export Options")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    "📥 Download as CSV",
                    csv,
                    "survey_results.csv",
                    "text/csv",
                    use_container_width=True
                )
            
            with col2:
                json_str = filtered_df.to_json(orient='records', indent=2)
                st.download_button(
                    "📥 Download as JSON",
                    json_str,
                    "survey_results.json",
                    "application/json",
                    use_container_width=True
                )
            
            with col3:
                # Excel download with metadata
                from io import BytesIO
                import pandas as pd
                
                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    filtered_df.to_excel(writer, index=False, sheet_name='Results')
                    
                    # Add metadata sheet
                    meta_df = pd.DataFrame([
                        {"Metric": k, "Value": str(v)}
                        for k, v in metadata.items()
                        if k != 'errors'
                    ])
                    meta_df.to_excel(writer, index=False, sheet_name='Metadata')
                
                excel_data = output.getvalue()
                st.download_button(
                    "📥 Download as Excel",
                    excel_data,
                    "survey_results.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            
            # Error log if any
            if metadata.get('errors'):
                with st.expander(f"⚠️ Error Log ({len(metadata['errors'])} errors)", expanded=False):
                    for error in metadata['errors']:
                        st.error(f"**{error['timestamp']}** - {error['model']}: {error['message']}")
        
        else:
            # No results available
            st.info("📭 No survey results available")
            st.markdown("""
                **To generate results:**
                1. Navigate to the Survey Executor page
                2. Configure your survey parameters
                3. Click "Start Survey Execution"
                4. Wait for completion
                5. Return here to view and export results
            """)
            
            if st.button("Go to Survey Executor", type="primary"):
                st.session_state['selected_page'] = "🚀 Survey Executor"
                st.rerun()
    
    elif page == "📚 Documentation":
        st.header("📚 Survey Executor Documentation")
        
        st.markdown("""
        ## Overview
        
        The Survey Executor is a sophisticated interface for running psychological and political 
        surveys across multiple Large Language Models with real-time monitoring and control.
        
        ## Features
        
        ### Pre-Execution
        - **Configuration Summary**: Review selected models, scales, and prompts
        - **Cost Estimation**: Get min/max cost estimates based on token usage
        - **Time Estimation**: Estimated completion time with rate limiting
        - **Task Calculation**: Total number of API calls to be made
        - **Warnings**: Alerts for high costs or long execution times
        
        ### During Execution
        - **Real-time Progress**: Overall and per-model progress bars
        - **Live Statistics**: Task completion, token usage, cost accumulation
        - **Current Status**: Shows which model/scale/prompt is being processed
        - **Error Tracking**: Live error count with expandable details
        - **Retry Monitoring**: Tracks automatic retry attempts
        - **Pause/Resume**: Pause execution and resume later
        - **Cancel with Save**: Cancel while saving partial results
        
        ### Post-Execution
        - **Summary Statistics**: Response rate, total cost, duration
        - **Results Preview**: Quick view of collected responses
        - **Multiple Export Formats**: CSV, JSON, Excel with metadata
        - **Error Analysis**: Review all errors encountered
        - **Continue to Analysis**: Seamless transition to analysis tools
        
        ## Usage Guide
        
        ### Step 1: Configuration
        1. Select models with configured API keys
        2. Choose psychological scales to administer
        3. Select prompt styles (political personas)
        4. Set number of runs and temperature
        
        ### Step 2: Review Estimates
        1. Check total task count
        2. Review cost estimates
        3. Note estimated completion time
        4. Address any warnings
        
        ### Step 3: Execute Survey
        1. Click "Start Survey Execution"
        2. Monitor real-time progress
        3. Use pause/resume if needed
        4. Save partial results if desired
        
        ### Step 4: Export Results
        1. Review summary statistics
        2. Filter results as needed
        3. Export in preferred format
        4. Continue to analysis
        
        ## Best Practices
        
        - **Start Small**: Test with 1-2 models and scales first
        - **Monitor Costs**: Keep an eye on the cost accumulator
        - **Handle Errors**: Review error logs to identify issues
        - **Save Frequently**: Use partial save for long runs
        - **Rate Limits**: The system handles rate limiting automatically
        
        ## Troubleshooting
        
        **High Failure Rate**
        - Check API key validity
        - Review error messages for rate limit issues
        - Ensure stable internet connection
        
        **Slow Execution**
        - Normal due to rate limiting
        - OpenAI: 3 concurrent, 1s delay
        - Claude: 5 concurrent, 0.5s delay
        - Llama: 10 concurrent, 0.2s delay
        
        **Cost Concerns**
        - Use fewer runs per question
        - Select fewer models
        - Reduce number of scales
        """)
    
    elif page == "🔧 Settings":
        st.header("🔧 Settings & Configuration")
        
        st.subheader("Execution Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.number_input(
                "Max Concurrent Calls",
                min_value=1,
                max_value=20,
                value=5,
                help="Maximum number of concurrent API calls"
            )
            
            st.number_input(
                "Retry Attempts",
                min_value=1,
                max_value=5,
                value=3,
                help="Number of retry attempts for failed calls"
            )
        
        with col2:
            st.number_input(
                "Request Timeout (seconds)",
                min_value=10,
                max_value=120,
                value=30,
                help="Timeout for individual API requests"
            )
            
            st.selectbox(
                "Progress Update Frequency",
                ["Real-time", "Every 5 tasks", "Every 10 tasks"],
                help="How often to update progress indicators"
            )
        
        st.subheader("Display Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.checkbox("Show detailed error messages", value=True)
            st.checkbox("Auto-expand error logs", value=False)
            st.checkbox("Show token usage in real-time", value=True)
        
        with col2:
            st.checkbox("Play sound on completion", value=False)
            st.checkbox("Show desktop notification", value=False)
            st.checkbox("Auto-save partial results", value=True)
        
        st.subheader("Export Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.selectbox(
                "Default Export Format",
                ["CSV", "JSON", "Excel"],
                help="Preferred format for data export"
            )
            
            st.checkbox("Include metadata in exports", value=True)
        
        with col2:
            st.text_input(
                "Export Directory",
                value="data/exports",
                help="Directory for saving exported files"
            )
            
            st.checkbox("Timestamp filenames", value=True)
        
        if st.button("💾 Save Settings", type="primary", use_container_width=True):
            st.success("Settings saved successfully!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #666;'>
            <small>
                LLM Survey Executor v1.0 | Real-time Progress Tracking | Cost Monitoring | Error Handling
            </small>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()