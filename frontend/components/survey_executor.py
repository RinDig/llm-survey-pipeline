"""Survey Execution Interface with Real-time Progress Tracking

This component provides a comprehensive execution and monitoring interface for running
LLM surveys with real-time progress updates, cost tracking, and error handling.
"""

import streamlit as st
import asyncio
import pandas as pd
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
from collections import defaultdict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SurveyExecutor:
    """Manages survey execution with real-time progress tracking and monitoring."""
    
    # Cost estimates per 1000 tokens (approximate pricing)
    TOKEN_COSTS = {
        "OpenAI": {"input": 0.003, "output": 0.006},  # GPT-4
        "Claude": {"input": 0.003, "output": 0.015},  # Claude 3.5 Sonnet
        "Llama": {"input": 0.0005, "output": 0.0015},  # Llama 3.1
        "Grok": {"input": 0.002, "output": 0.004},  # Grok 2
        "DeepSeek": {"input": 0.0004, "output": 0.0012},  # DeepSeek v3
    }
    
    # Average tokens per question/response
    AVG_TOKENS_PER_QUESTION = 150
    AVG_TOKENS_PER_RESPONSE = 100
    
    def __init__(self):
        """Initialize the survey executor."""
        self.initialize_session_state()
        
    def initialize_session_state(self):
        """Initialize session state variables for execution tracking."""
        if 'execution_state' not in st.session_state:
            st.session_state.execution_state = {
                'status': 'idle',  # idle, running, paused, completed, cancelled
                'start_time': None,
                'end_time': None,
                'progress': 0,
                'total_tasks': 0,
                'completed_tasks': 0,
                'failed_tasks': 0,
                'current_model': None,
                'current_scale': None,
                'current_prompt': None,
                'errors': [],
                'token_usage': defaultdict(int),
                'cost_accumulator': 0.0,
                'results': None,
                'pause_requested': False,
                'cancel_requested': False,
                'model_progress': {},
                'retry_counts': defaultdict(int),
            }
    
    def calculate_cost_estimate(self, config: Dict) -> Tuple[float, float]:
        """Calculate estimated cost range for the survey configuration.
        
        Args:
            config: Survey configuration dictionary
            
        Returns:
            Tuple of (min_cost, max_cost) estimates
        """
        total_questions = len(config.get('questions', []))
        num_models = len(config.get('models', []))
        num_prompts = len(config.get('prompts', []))
        num_runs = config.get('runs', 1)
        
        total_tasks = total_questions * num_models * num_prompts * num_runs
        
        min_cost = 0
        max_cost = 0
        
        for model in config.get('models', []):
            if model in self.TOKEN_COSTS:
                costs = self.TOKEN_COSTS[model]
                # Calculate per-task cost
                input_cost = (self.AVG_TOKENS_PER_QUESTION / 1000) * costs['input']
                output_cost = (self.AVG_TOKENS_PER_RESPONSE / 1000) * costs['output']
                task_cost = input_cost + output_cost
                
                # Calculate model's share of tasks
                model_tasks = (total_tasks / num_models) if num_models > 0 else 0
                
                min_cost += model_tasks * task_cost * 0.8  # 80% for minimum estimate
                max_cost += model_tasks * task_cost * 1.5  # 150% for maximum estimate
        
        return min_cost, max_cost
    
    def estimate_completion_time(self, config: Dict) -> timedelta:
        """Estimate time to complete the survey based on configuration.
        
        Args:
            config: Survey configuration dictionary
            
        Returns:
            Estimated timedelta for completion
        """
        total_questions = len(config.get('questions', []))
        num_models = len(config.get('models', []))
        num_prompts = len(config.get('prompts', []))
        num_runs = config.get('runs', 1)
        
        total_tasks = total_questions * num_models * num_prompts * num_runs
        
        # Estimate based on model response times (seconds per task)
        response_times = {
            "OpenAI": 2.0,
            "Claude": 1.5,
            "Llama": 1.0,
            "Grok": 2.5,
            "DeepSeek": 1.2,
        }
        
        avg_time = 0
        for model in config.get('models', []):
            avg_time += response_times.get(model, 2.0)
        
        if num_models > 0:
            avg_time = avg_time / num_models
        
        # Add overhead for rate limiting and retries
        total_seconds = total_tasks * avg_time * 1.2
        
        return timedelta(seconds=total_seconds)
    
    def render_pre_execution(self, config: Dict):
        """Render the pre-execution interface with configuration summary and estimates.
        
        Args:
            config: Survey configuration dictionary
        """
        st.subheader("📋 Configuration Summary")
        
        # Configuration details in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Selected Models", len(config.get('models', [])))
            with st.expander("Models List", expanded=False):
                for model in config.get('models', []):
                    st.write(f"• {model}")
        
        with col2:
            st.metric("Selected Scales", len(config.get('scales', [])))
            with st.expander("Scales List", expanded=False):
                for scale in config.get('scales', []):
                    st.write(f"• {scale}")
        
        with col3:
            st.metric("Prompt Styles", len(config.get('prompts', [])))
            with st.expander("Prompts List", expanded=False):
                for prompt in config.get('prompts', []):
                    st.write(f"• {prompt}")
        
        # Task calculation
        total_questions = len(config.get('questions', []))
        num_runs = config.get('runs', 1)
        total_tasks = (
            total_questions * 
            len(config.get('models', [])) * 
            len(config.get('prompts', [])) * 
            num_runs
        )
        
        st.markdown("---")
        
        # Estimates section
        st.subheader("📊 Execution Estimates")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Tasks", f"{total_tasks:,}")
            st.caption(f"{total_questions} questions × {len(config.get('models', []))} models × {len(config.get('prompts', []))} prompts × {num_runs} runs")
        
        with col2:
            min_cost, max_cost = self.calculate_cost_estimate(config)
            st.metric("Estimated Cost", f"${min_cost:.2f} - ${max_cost:.2f}")
            st.caption("Based on average token usage")
        
        with col3:
            time_est = self.estimate_completion_time(config)
            hours = time_est.total_seconds() / 3600
            if hours < 1:
                time_str = f"{int(time_est.total_seconds() / 60)} minutes"
            else:
                time_str = f"{hours:.1f} hours"
            st.metric("Estimated Time", time_str)
            st.caption("Including rate limiting")
        
        with col4:
            st.metric("Temperature", config.get('temperature', 0.0))
            st.caption("Response randomness")
        
        # Warnings for high costs or long runs
        if max_cost > 50:
            st.warning(f"⚠️ High estimated cost: ${max_cost:.2f}. Please review your configuration.")
        
        if time_est.total_seconds() > 3600:
            st.warning(f"⚠️ Long execution time: {time_str}. Consider reducing the number of runs or models.")
        
        if total_tasks > 1000:
            st.info(f"ℹ️ Large survey: {total_tasks:,} tasks will be processed. The interface will update progress in real-time.")
        
        # Execution controls
        st.markdown("---")
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            if st.button("🚀 Start Survey Execution", type="primary", use_container_width=True,
                        disabled=st.session_state.execution_state['status'] == 'running'):
                st.session_state.execution_state['status'] = 'running'
                st.session_state.execution_state['start_time'] = datetime.now()
                st.session_state.execution_state['total_tasks'] = total_tasks
                st.rerun()
        
        with col2:
            if st.button("❌ Cancel", type="secondary", use_container_width=True,
                        disabled=st.session_state.execution_state['status'] != 'idle'):
                st.session_state.execution_state = self.initialize_session_state()
        
        with col3:
            if st.button("🔄 Reset", use_container_width=True):
                st.session_state.execution_state = self.initialize_session_state()
                st.rerun()
    
    def render_execution_progress(self):
        """Render the real-time execution progress interface."""
        st.subheader("🔄 Survey Execution in Progress")
        
        # Overall progress
        progress = st.session_state.execution_state['progress']
        total_tasks = st.session_state.execution_state['total_tasks']
        completed_tasks = st.session_state.execution_state['completed_tasks']
        
        # Main progress bar
        progress_pct = (completed_tasks / total_tasks) if total_tasks > 0 else 0
        st.progress(progress_pct, text=f"Overall Progress: {completed_tasks}/{total_tasks} tasks ({progress_pct*100:.1f}%)")
        
        # Execution stats in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            elapsed = datetime.now() - st.session_state.execution_state['start_time']
            st.metric("Elapsed Time", f"{elapsed.seconds // 60}m {elapsed.seconds % 60}s")
        
        with col2:
            st.metric("Completed Tasks", f"{completed_tasks:,}")
        
        with col3:
            failed = st.session_state.execution_state['failed_tasks']
            color = "🔴" if failed > 0 else "🟢"
            st.metric("Failed Tasks", f"{color} {failed}")
        
        with col4:
            cost = st.session_state.execution_state['cost_accumulator']
            st.metric("Current Cost", f"${cost:.3f}")
        
        st.markdown("---")
        
        # Current processing status
        with st.container():
            st.subheader("📍 Current Processing")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                current_model = st.session_state.execution_state.get('current_model', 'N/A')
                st.info(f"**Model:** {current_model}")
            
            with col2:
                current_scale = st.session_state.execution_state.get('current_scale', 'N/A')
                st.info(f"**Scale:** {current_scale}")
            
            with col3:
                current_prompt = st.session_state.execution_state.get('current_prompt', 'N/A')
                st.info(f"**Prompt:** {current_prompt}")
        
        # Per-model progress
        st.markdown("---")
        st.subheader("📊 Model Progress")
        
        model_progress = st.session_state.execution_state.get('model_progress', {})
        if model_progress:
            cols = st.columns(len(model_progress))
            for idx, (model, progress_data) in enumerate(model_progress.items()):
                with cols[idx]:
                    completed = progress_data.get('completed', 0)
                    total = progress_data.get('total', 1)
                    pct = (completed / total) if total > 0 else 0
                    st.metric(model, f"{completed}/{total}")
                    st.progress(pct)
        
        # Token usage tracking
        st.markdown("---")
        st.subheader("🎫 Token Usage")
        
        token_usage = st.session_state.execution_state.get('token_usage', {})
        if token_usage:
            cols = st.columns(len(token_usage))
            for idx, (model, tokens) in enumerate(token_usage.items()):
                with cols[idx]:
                    st.metric(f"{model} Tokens", f"{tokens:,}")
        
        # Error tracking
        errors = st.session_state.execution_state.get('errors', [])
        if errors:
            with st.expander(f"⚠️ Errors ({len(errors)})", expanded=False):
                for error in errors[-10:]:  # Show last 10 errors
                    st.error(f"**{error['timestamp']}** - {error['model']}: {error['message']}")
        
        # Retry tracking
        retry_counts = st.session_state.execution_state.get('retry_counts', {})
        if any(retry_counts.values()):
            with st.expander("🔄 Retry Statistics", expanded=False):
                for model, count in retry_counts.items():
                    if count > 0:
                        st.write(f"**{model}:** {count} retries")
        
        # Control buttons
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.session_state.execution_state['status'] == 'running':
                if st.button("⏸️ Pause", use_container_width=True):
                    st.session_state.execution_state['pause_requested'] = True
                    st.session_state.execution_state['status'] = 'paused'
            elif st.session_state.execution_state['status'] == 'paused':
                if st.button("▶️ Resume", use_container_width=True):
                    st.session_state.execution_state['pause_requested'] = False
                    st.session_state.execution_state['status'] = 'running'
        
        with col2:
            if st.button("🛑 Cancel", type="secondary", use_container_width=True):
                st.session_state.execution_state['cancel_requested'] = True
                if st.session_state.execution_state['results']:
                    st.info("Saving partial results...")
                    self._save_partial_results()
        
        with col3:
            if st.button("💾 Save Partial", use_container_width=True):
                if st.session_state.execution_state['results']:
                    self._save_partial_results()
                    st.success("Partial results saved!")
    
    def render_post_execution(self):
        """Render the post-execution summary and results interface."""
        st.subheader("✅ Survey Execution Complete")
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total = st.session_state.execution_state['total_tasks']
            completed = st.session_state.execution_state['completed_tasks']
            response_rate = (completed / total * 100) if total > 0 else 0
            st.metric("Response Rate", f"{response_rate:.1f}%")
            st.caption(f"{completed}/{total} tasks completed")
        
        with col2:
            cost = st.session_state.execution_state['cost_accumulator']
            st.metric("Total Cost", f"${cost:.2f}")
            
        with col3:
            duration = st.session_state.execution_state['end_time'] - st.session_state.execution_state['start_time']
            st.metric("Duration", f"{duration.seconds // 60}m {duration.seconds % 60}s")
        
        with col4:
            failed = st.session_state.execution_state['failed_tasks']
            color = "🔴" if failed > 0 else "🟢"
            st.metric("Failed Tasks", f"{color} {failed}")
        
        st.markdown("---")
        
        # Token usage summary
        st.subheader("📊 Token Usage Summary")
        token_usage = st.session_state.execution_state.get('token_usage', {})
        
        if token_usage:
            # Create a DataFrame for better visualization
            token_df = pd.DataFrame([
                {"Model": model, "Tokens Used": tokens, "Estimated Cost": self._calculate_model_cost(model, tokens)}
                for model, tokens in token_usage.items()
            ])
            st.dataframe(token_df, use_container_width=True, hide_index=True)
        
        # Results preview
        st.markdown("---")
        st.subheader("📋 Results Preview")
        
        results = st.session_state.execution_state.get('results')
        if results and isinstance(results, pd.DataFrame):
            st.info(f"Total responses collected: {len(results)}")
            
            # Show first few rows
            with st.expander("View Sample Results", expanded=True):
                st.dataframe(results.head(10), use_container_width=True)
            
            # Basic statistics
            if 'scored_value' in results.columns:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Mean Score", f"{results['scored_value'].mean():.2f}")
                with col2:
                    st.metric("Std Deviation", f"{results['scored_value'].std():.2f}")
        
        # Save options
        st.markdown("---")
        st.subheader("💾 Save Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"survey_results_{timestamp}"
            
            if st.button("📥 Download CSV", use_container_width=True):
                if results is not None:
                    csv = results.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"{filename}.csv",
                        mime="text/csv"
                    )
        
        with col2:
            if st.button("📥 Download JSON", use_container_width=True):
                if results is not None:
                    json_str = results.to_json(orient='records', indent=2)
                    st.download_button(
                        label="Download JSON",
                        data=json_str,
                        file_name=f"{filename}.json",
                        mime="application/json"
                    )
        
        with col3:
            if st.button("📥 Download Excel", use_container_width=True):
                if results is not None:
                    # Create Excel file in memory
                    from io import BytesIO
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        results.to_excel(writer, index=False, sheet_name='Survey Results')
                        
                        # Add metadata sheet
                        metadata = pd.DataFrame([
                            {"Metric": "Total Tasks", "Value": st.session_state.execution_state['total_tasks']},
                            {"Metric": "Completed Tasks", "Value": st.session_state.execution_state['completed_tasks']},
                            {"Metric": "Failed Tasks", "Value": st.session_state.execution_state['failed_tasks']},
                            {"Metric": "Total Cost", "Value": f"${st.session_state.execution_state['cost_accumulator']:.2f}"},
                            {"Metric": "Duration", "Value": str(st.session_state.execution_state['end_time'] - st.session_state.execution_state['start_time'])},
                        ])
                        metadata.to_excel(writer, index=False, sheet_name='Metadata')
                    
                    excel_data = output.getvalue()
                    st.download_button(
                        label="Download Excel",
                        data=excel_data,
                        file_name=f"{filename}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
        
        # Action buttons
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Continue to Analysis", type="primary", use_container_width=True):
                st.session_state['analysis_data'] = results
                st.info("Results saved to session. Navigate to the Analysis tab.")
        
        with col2:
            if st.button("🔄 Run New Survey", use_container_width=True):
                # Reset execution state but keep configuration
                st.session_state.execution_state = self.initialize_session_state()
                st.rerun()
        
        with col3:
            if st.button("🗑️ Clear Results", type="secondary", use_container_width=True):
                st.session_state.execution_state['results'] = None
                st.session_state.execution_state['status'] = 'idle'
                st.rerun()
    
    def _calculate_model_cost(self, model: str, tokens: int) -> float:
        """Calculate estimated cost for a model based on token usage.
        
        Args:
            model: Model name
            tokens: Total tokens used
            
        Returns:
            Estimated cost in dollars
        """
        if model in self.TOKEN_COSTS:
            costs = self.TOKEN_COSTS[model]
            # Assume 60% input, 40% output split
            input_tokens = tokens * 0.6
            output_tokens = tokens * 0.4
            
            input_cost = (input_tokens / 1000) * costs['input']
            output_cost = (output_tokens / 1000) * costs['output']
            
            return input_cost + output_cost
        return 0.0
    
    def _save_partial_results(self):
        """Save partial results to session state and optionally to file."""
        results = st.session_state.execution_state.get('results')
        if results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.session_state[f'partial_results_{timestamp}'] = results
            return True
        return False
    
    async def execute_survey_async(self, pipeline, config: Dict):
        """Execute the survey asynchronously with progress callbacks.
        
        Args:
            pipeline: SurveyPipeline instance
            config: Survey configuration dictionary
        """
        try:
            # Build tasks
            tasks = pipeline.build_tasks()
            st.session_state.execution_state['total_tasks'] = len(tasks)
            
            # Initialize model progress tracking
            model_counts = defaultdict(int)
            for task in tasks:
                model_counts[task['model_name']] += 1
            
            for model, count in model_counts.items():
                st.session_state.execution_state['model_progress'][model] = {
                    'total': count,
                    'completed': 0
                }
            
            # Custom progress callback
            def progress_callback(task_info, result):
                """Update progress in session state."""
                st.session_state.execution_state['completed_tasks'] += 1
                st.session_state.execution_state['current_model'] = task_info.get('model_name')
                st.session_state.execution_state['current_scale'] = task_info.get('scale_name')
                st.session_state.execution_state['current_prompt'] = task_info.get('prompt_style')
                
                # Update model progress
                model = task_info.get('model_name')
                if model in st.session_state.execution_state['model_progress']:
                    st.session_state.execution_state['model_progress'][model]['completed'] += 1
                
                # Update progress percentage
                completed = st.session_state.execution_state['completed_tasks']
                total = st.session_state.execution_state['total_tasks']
                st.session_state.execution_state['progress'] = (completed / total) if total > 0 else 0
                
                # Track errors
                if isinstance(result, Exception) or (isinstance(result, dict) and 'ERROR' in result.get('justification', '')):
                    error_entry = {
                        'timestamp': datetime.now().strftime("%H:%M:%S"),
                        'model': task_info.get('model_name'),
                        'message': str(result) if isinstance(result, Exception) else result.get('justification', 'Unknown error')
                    }
                    st.session_state.execution_state['errors'].append(error_entry)
                    st.session_state.execution_state['failed_tasks'] += 1
                    
                    # Track retries
                    model = task_info.get('model_name')
                    st.session_state.execution_state['retry_counts'][model] += 1
            
            # Execute with modified processor that includes callbacks
            results = await self._process_with_callbacks(tasks, pipeline, progress_callback)
            
            # Store results
            df_results = pd.DataFrame(results)
            df_results["scored_value"] = df_results.apply(pipeline.apply_reverse_score, axis=1)
            
            st.session_state.execution_state['results'] = df_results
            st.session_state.execution_state['status'] = 'completed'
            st.session_state.execution_state['end_time'] = datetime.now()
            
            return df_results
            
        except Exception as e:
            logger.error(f"Survey execution failed: {str(e)}")
            st.session_state.execution_state['status'] = 'error'
            st.session_state.execution_state['errors'].append({
                'timestamp': datetime.now().strftime("%H:%M:%S"),
                'model': 'System',
                'message': str(e)
            })
            raise
    
    async def _process_with_callbacks(self, tasks, pipeline, callback):
        """Process tasks with progress callbacks.
        
        This is a modified version of process_tasks_in_chunks that includes callbacks.
        """
        # Import necessary modules
        from backend.core.api_clients import call_model_api
        from backend.utils.cost_tracking import cost_tracker
        
        results = []
        chunk_size = 5
        
        # Sort tasks by API provider
        openai_queue = []
        anthropic_queue = []
        llama_queue = []
        
        for task in tasks:
            if task["model_name"] in ["OpenAI", "Grok"]:
                openai_queue.append(task)
            elif task["model_name"] == "Claude":
                anthropic_queue.append(task)
            else:
                llama_queue.append(task)
        
        async def process_queue_with_callback(queue, semaphore, rate_limit):
            """Process a queue with callbacks."""
            queue_results = []
            for i in range(0, len(queue), chunk_size):
                # Check for pause/cancel
                while st.session_state.execution_state.get('pause_requested', False):
                    await asyncio.sleep(0.5)
                
                if st.session_state.execution_state.get('cancel_requested', False):
                    break
                
                chunk = queue[i:i + chunk_size]
                async with semaphore:
                    coros = [
                        call_model_api(
                            t["model_name"],
                            t["question_text"],
                            t["prompt_style"],
                            t["scale_range"],
                            0.0
                        )
                        for t in chunk
                    ]
                    chunk_results = await asyncio.gather(*coros, return_exceptions=True)
                    
                    for task_info, result in zip(chunk, chunk_results):
                        # Call the progress callback
                        callback(task_info, result)
                        
                        # Update token usage
                        if task_info["model_name"] in cost_tracker:
                            st.session_state.execution_state['token_usage'][task_info["model_name"]] = cost_tracker[task_info["model_name"]]
                            
                            # Update cost accumulator
                            tokens = cost_tracker[task_info["model_name"]]
                            cost = self._calculate_model_cost(task_info["model_name"], tokens)
                            st.session_state.execution_state['cost_accumulator'] = cost
                        
                        queue_results.append((task_info, result))
                    
                    await asyncio.sleep(rate_limit)
            
            return queue_results
        
        # Create semaphores
        openai_sem = asyncio.Semaphore(3)
        anthropic_sem = asyncio.Semaphore(5)
        llama_sem = asyncio.Semaphore(10)
        
        # Process all queues
        queue_tasks = [
            process_queue_with_callback(openai_queue, openai_sem, 1.0),
            process_queue_with_callback(anthropic_queue, anthropic_sem, 0.5),
            process_queue_with_callback(llama_queue, llama_sem, 0.2),
        ]
        
        all_results = await asyncio.gather(*queue_tasks)
        
        # Process results
        for queue_result in all_results:
            for task_info, result in queue_result:
                if isinstance(result, Exception):
                    min_scale, max_scale = task_info["scale_range"]
                    midpoint = (min_scale + max_scale) / 2
                    result_dict = {
                        "model_name": task_info["model_name"],
                        "numeric_score": midpoint,
                        "label": None,
                        "justification": f"ERROR: {str(result)}",
                        "raw_response": str(result),
                        "duration": None
                    }
                else:
                    result_dict = result
                
                # Validate numeric score
                min_scale, max_scale = task_info["scale_range"]
                if not (min_scale <= result_dict["numeric_score"] <= max_scale):
                    result_dict["numeric_score"] = (min_scale + max_scale) / 2
                    result_dict["justification"] = f"RANGE ERROR: Original score: {result_dict['numeric_score']}"
                
                result_dict.update(task_info)
                results.append(result_dict)
        
        return results
    
    def render(self, config: Dict):
        """Main render method for the survey executor component.
        
        Args:
            config: Survey configuration dictionary containing:
                - models: List of model names
                - scales: List of scale names
                - prompts: List of prompt styles
                - questions: List of question dictionaries
                - runs: Number of runs per question
                - temperature: Temperature setting
        """
        status = st.session_state.execution_state['status']
        
        if status == 'idle':
            self.render_pre_execution(config)
        elif status in ['running', 'paused']:
            self.render_execution_progress()
            
            # Auto-refresh for progress updates
            if status == 'running':
                time.sleep(1)
                st.rerun()
        elif status == 'completed':
            self.render_post_execution()
        elif status == 'error':
            st.error("❌ Survey execution encountered an error. Please check the error log.")
            errors = st.session_state.execution_state.get('errors', [])
            if errors:
                with st.expander("Error Details", expanded=True):
                    for error in errors:
                        st.error(f"**{error['timestamp']}** - {error['model']}: {error['message']}")
            
            if st.button("🔄 Reset", type="primary"):
                st.session_state.execution_state = self.initialize_session_state()
                st.rerun()


def render_survey_executor(config: Optional[Dict] = None):
    """Convenience function to render the survey executor component.
    
    Args:
        config: Optional survey configuration dictionary. If not provided,
                will attempt to get from session state.
    """
    executor = SurveyExecutor()
    
    # Get config from session state if not provided
    if config is None:
        config = st.session_state.get('survey_config', {})
    
    # Validate config has required fields
    if not config:
        st.warning("⚠️ No survey configuration found. Please configure your survey first.")
        return
    
    required_fields = ['models', 'scales', 'prompts', 'questions']
    missing_fields = [field for field in required_fields if field not in config]
    
    if missing_fields:
        st.error(f"❌ Missing required configuration fields: {', '.join(missing_fields)}")
        return
    
    # Render the executor interface
    executor.render(config)
    
    # Handle async execution if running
    if st.session_state.execution_state['status'] == 'running' and 'pipeline' in st.session_state:
        pipeline = st.session_state['pipeline']
        asyncio.run(executor.execute_survey_async(pipeline, config))


# Example usage for testing
if __name__ == "__main__":
    st.set_page_config(
        page_title="Survey Executor Test",
        page_icon="🚀",
        layout="wide"
    )
    
    st.title("🚀 Survey Executor Component Test")
    
    # Create sample configuration
    sample_config = {
        'models': ['OpenAI', 'Claude'],
        'scales': ['RWA', 'LWA'],
        'prompts': ['minimal', 'extreme_liberal'],
        'questions': [
            {'id': 'q1', 'text': 'Sample question 1', 'scale_name': 'RWA', 'scale_range': [1, 7]},
            {'id': 'q2', 'text': 'Sample question 2', 'scale_name': 'LWA', 'scale_range': [1, 7]},
        ],
        'runs': 2,
        'temperature': 0.0
    }
    
    # Store in session state
    st.session_state['survey_config'] = sample_config
    
    # Render the executor
    render_survey_executor()