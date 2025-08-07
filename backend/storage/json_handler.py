"""JSON storage handler for survey results with metadata"""
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
import hashlib


class SurveyResultStorage:
    """Handles storage and retrieval of survey results with metadata"""
    
    def __init__(self, storage_path: str = "data/storage"):
        """Initialize storage handler
        
        Args:
            storage_path: Base path for storing JSON files
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.index_file = self.storage_path / "index.json"
        self._load_index()
    
    def _load_index(self):
        """Load or create the storage index"""
        if self.index_file.exists():
            with open(self.index_file, 'r') as f:
                self.index = json.load(f)
        else:
            self.index = {"runs": [], "total_runs": 0, "last_updated": None}
            self._save_index()
    
    def _save_index(self):
        """Save the storage index"""
        self.index["last_updated"] = datetime.now().isoformat()
        with open(self.index_file, 'w') as f:
            json.dump(self.index, f, indent=2)
    
    def save_survey_results(
        self,
        results_df: pd.DataFrame,
        configuration: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        costs: Optional[Dict[str, Any]] = None,
        errors: Optional[List[Dict]] = None
    ) -> str:
        """Save survey results with comprehensive metadata
        
        Args:
            results_df: DataFrame containing survey responses
            configuration: Survey configuration (models, scales, prompts)
            metadata: Optional metadata (researcher info, tags, etc.)
            costs: Token usage and cost information
            errors: List of errors encountered during execution
            
        Returns:
            run_id: Unique identifier for this run
        """
        # Generate unique run ID
        run_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        # Prepare results data
        results_data = results_df.to_dict(orient='records') if not results_df.empty else []
        
        # Calculate statistics
        statistics = self._calculate_statistics(results_df) if not results_df.empty else {}
        
        # Create comprehensive data structure
        survey_data = {
            "run_id": run_id,
            "timestamp": timestamp,
            "metadata": metadata or {},
            "configuration": configuration,
            "results": results_data,
            "costs": costs or {},
            "statistics": statistics,
            "errors": errors or [],
            "summary": {
                "total_responses": len(results_data),
                "models_used": configuration.get("models", []),
                "scales_used": configuration.get("scales", []),
                "prompts_used": configuration.get("prompts", []),
                "success_rate": statistics.get("response_rate", 0)
            }
        }
        
        # Add checksum for data integrity
        survey_data["checksum"] = self._calculate_checksum(survey_data)
        
        # Save to file
        file_path = self.storage_path / f"survey_{run_id}.json"
        with open(file_path, 'w') as f:
            json.dump(survey_data, f, indent=2, default=str)
        
        # Update index
        index_entry = {
            "run_id": run_id,
            "timestamp": timestamp,
            "file_path": str(file_path),
            "metadata": metadata or {},
            "summary": survey_data["summary"],
            "tags": metadata.get("tags", []) if metadata else []
        }
        self.index["runs"].append(index_entry)
        self.index["total_runs"] += 1
        self._save_index()
        
        return run_id
    
    def load_survey_results(self, run_id: str) -> Optional[Dict]:
        """Load survey results by run ID
        
        Args:
            run_id: Unique identifier for the run
            
        Returns:
            Survey data dictionary or None if not found
        """
        # Find in index
        for run in self.index["runs"]:
            if run["run_id"] == run_id:
                file_path = Path(run["file_path"])
                if file_path.exists():
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    # Verify checksum
                    stored_checksum = data.pop("checksum", None)
                    calculated_checksum = self._calculate_checksum(data)
                    if stored_checksum != calculated_checksum:
                        print(f"Warning: Checksum mismatch for run {run_id}")
                    data["checksum"] = stored_checksum
                    
                    return data
        return None
    
    def search_runs(
        self,
        filters: Optional[Dict[str, Any]] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[Dict]:
        """Search for runs based on filters
        
        Args:
            filters: Dictionary of filters (e.g., {"model": "OpenAI"})
            date_from: Start date (ISO format)
            date_to: End date (ISO format)
            tags: List of tags to filter by
            
        Returns:
            List of matching run summaries
        """
        results = []
        
        for run in self.index["runs"]:
            # Date filtering
            if date_from and run["timestamp"] < date_from:
                continue
            if date_to and run["timestamp"] > date_to:
                continue
            
            # Tag filtering
            if tags:
                run_tags = run.get("tags", [])
                if not any(tag in run_tags for tag in tags):
                    continue
            
            # Custom filters
            if filters:
                match = True
                for key, value in filters.items():
                    if key in run["summary"]:
                        if isinstance(value, list):
                            if not any(v in run["summary"][key] for v in value):
                                match = False
                                break
                        elif run["summary"].get(key) != value:
                            match = False
                            break
                if not match:
                    continue
            
            results.append(run)
        
        return sorted(results, key=lambda x: x["timestamp"], reverse=True)
    
    def get_all_runs(self) -> List[Dict]:
        """Get all run summaries
        
        Returns:
            List of all run summaries
        """
        return sorted(self.index["runs"], key=lambda x: x["timestamp"], reverse=True)
    
    def delete_run(self, run_id: str) -> bool:
        """Delete a survey run
        
        Args:
            run_id: Unique identifier for the run
            
        Returns:
            True if deleted successfully, False otherwise
        """
        for i, run in enumerate(self.index["runs"]):
            if run["run_id"] == run_id:
                # Delete file
                file_path = Path(run["file_path"])
                if file_path.exists():
                    file_path.unlink()
                
                # Remove from index
                self.index["runs"].pop(i)
                self._save_index()
                return True
        return False
    
    def export_combined_results(
        self,
        run_ids: List[str],
        output_path: str,
        format: str = "json"
    ) -> bool:
        """Export combined results from multiple runs
        
        Args:
            run_ids: List of run IDs to combine
            output_path: Path for output file
            format: Export format ('json', 'csv', 'excel')
            
        Returns:
            True if export successful
        """
        combined_data = []
        combined_metadata = {
            "export_timestamp": datetime.now().isoformat(),
            "included_runs": run_ids,
            "total_responses": 0
        }
        
        for run_id in run_ids:
            data = self.load_survey_results(run_id)
            if data:
                # Add run_id to each result
                for result in data.get("results", []):
                    result["run_id"] = run_id
                    result["run_timestamp"] = data["timestamp"]
                    combined_data.append(result)
                combined_metadata["total_responses"] += len(data.get("results", []))
        
        if not combined_data:
            return False
        
        output_path = Path(output_path)
        
        if format == "json":
            export_data = {
                "metadata": combined_metadata,
                "results": combined_data
            }
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
        
        elif format == "csv":
            df = pd.DataFrame(combined_data)
            df.to_csv(output_path, index=False)
        
        elif format == "excel":
            df = pd.DataFrame(combined_data)
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Results', index=False)
                
                # Add metadata sheet
                meta_df = pd.DataFrame([combined_metadata])
                meta_df.to_excel(writer, sheet_name='Metadata', index=False)
        
        return True
    
    def _calculate_statistics(self, df: pd.DataFrame) -> Dict:
        """Calculate statistics from results DataFrame
        
        Args:
            df: Results DataFrame
            
        Returns:
            Dictionary of statistics
        """
        stats = {}
        
        if not df.empty:
            # Response rate
            total_questions = len(df)
            valid_responses = df['numeric_score'].notna().sum()
            stats['response_rate'] = (valid_responses / total_questions * 100) if total_questions > 0 else 0
            
            # Per-model statistics
            if 'model' in df.columns:
                model_stats = {}
                for model in df['model'].unique():
                    model_df = df[df['model'] == model]
                    model_stats[model] = {
                        'total': len(model_df),
                        'valid': model_df['numeric_score'].notna().sum(),
                        'mean_score': model_df['numeric_score'].mean() if not model_df['numeric_score'].isna().all() else None
                    }
                stats['model_statistics'] = model_stats
            
            # Per-scale statistics
            if 'scale_name' in df.columns:
                scale_stats = {}
                for scale in df['scale_name'].unique():
                    scale_df = df[df['scale_name'] == scale]
                    scale_stats[scale] = {
                        'total': len(scale_df),
                        'valid': scale_df['numeric_score'].notna().sum(),
                        'mean_score': scale_df['numeric_score'].mean() if not scale_df['numeric_score'].isna().all() else None
                    }
                stats['scale_statistics'] = scale_stats
            
            # Overall statistics
            stats['total_questions'] = total_questions
            stats['valid_responses'] = int(valid_responses)
            stats['overall_mean'] = df['numeric_score'].mean() if not df['numeric_score'].isna().all() else None
            stats['overall_std'] = df['numeric_score'].std() if not df['numeric_score'].isna().all() else None
        
        return stats
    
    def _calculate_checksum(self, data: Dict) -> str:
        """Calculate checksum for data integrity
        
        Args:
            data: Data dictionary
            
        Returns:
            SHA256 checksum
        """
        # Convert to string for hashing
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()


class StorageManager:
    """High-level storage management interface"""
    
    def __init__(self, storage_path: str = "data/storage"):
        self.storage = SurveyResultStorage(storage_path)
    
    def save_from_pipeline(
        self,
        results_df: pd.DataFrame,
        pipeline_config: Dict,
        user_metadata: Optional[Dict] = None
    ) -> str:
        """Save results directly from pipeline execution
        
        Args:
            results_df: Results from pipeline
            pipeline_config: Configuration used for pipeline
            user_metadata: Additional metadata from user
            
        Returns:
            run_id for the saved results
        """
        # Extract costs from pipeline if available
        costs = pipeline_config.get("costs", {})
        
        # Extract errors if any
        errors = pipeline_config.get("errors", [])
        
        # Prepare configuration
        configuration = {
            "models": pipeline_config.get("models_to_run", []),
            "scales": pipeline_config.get("scales_to_run", []),
            "prompts": pipeline_config.get("prompt_styles_to_run", []),
            "temperature": pipeline_config.get("temperature", 0.0),
            "num_runs": pipeline_config.get("num_calls_test", 1)
        }
        
        # Save and return run_id
        return self.storage.save_survey_results(
            results_df,
            configuration,
            metadata=user_metadata,
            costs=costs,
            errors=errors
        )
    
    def get_recent_runs(self, limit: int = 10) -> List[Dict]:
        """Get recent run summaries
        
        Args:
            limit: Number of recent runs to return
            
        Returns:
            List of recent run summaries
        """
        all_runs = self.storage.get_all_runs()
        return all_runs[:limit]
    
    def search_by_model(self, model_name: str) -> List[Dict]:
        """Search for runs using a specific model
        
        Args:
            model_name: Name of the model
            
        Returns:
            List of matching runs
        """
        return self.storage.search_runs(filters={"models_used": [model_name]})
    
    def search_by_date_range(self, date_from: str, date_to: str) -> List[Dict]:
        """Search for runs within a date range
        
        Args:
            date_from: Start date (ISO format)
            date_to: End date (ISO format)
            
        Returns:
            List of matching runs
        """
        return self.storage.search_runs(date_from=date_from, date_to=date_to)