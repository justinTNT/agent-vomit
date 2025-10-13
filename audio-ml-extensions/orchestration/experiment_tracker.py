import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
import json
import time
import hashlib
import os
from datetime import datetime
from collections import defaultdict
import pickle
import warnings
import shutil


class Experiment:
    """Single experiment instance."""
    def __init__(self, 
                 name: str,
                 tags: Optional[List[str]] = None,
                 config: Optional[Dict[str, Any]] = None):
        self.id = self._generate_id()
        self.name = name
        self.tags = tags or []
        self.config = config or {}
        self.metrics = defaultdict(list)
        self.artifacts = {}
        self.checkpoints = []
        self.status = 'running'
        self.start_time = datetime.now()
        self.end_time = None
        self.metadata = {
            'created_at': self.start_time.isoformat(),
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
        
    def _generate_id(self) -> str:
        """Generate unique experiment ID."""
        timestamp = str(time.time()).encode()
        return hashlib.md5(timestamp).hexdigest()[:12]
        
    def duration(self) -> Optional[float]:
        """Get experiment duration in seconds."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None


class ExperimentTracker(nn.Module):
    def __init__(self,
                 project_name: str,
                 base_dir: str = './experiments',
                 auto_save: bool = True,
                 save_interval: int = 100,
                 track_gradients: bool = False,
                 track_parameters: bool = False,
                 version_control: bool = True):
        super().__init__()
        self.project_name = project_name
        self.base_dir = base_dir
        self.auto_save = auto_save
        self.save_interval = save_interval
        self.track_gradients = track_gradients
        self.track_parameters = track_parameters
        self.version_control = version_control
        
        # Setup directories
        self.project_dir = os.path.join(base_dir, project_name)
        os.makedirs(self.project_dir, exist_ok=True)
        
        # Tracking state
        self.current_experiment = None
        self.experiments = self._load_experiments()
        self.step = 0
        
    def _load_experiments(self) -> Dict[str, Experiment]:
        """Load existing experiments from disk."""
        experiments = {}
        experiments_file = os.path.join(self.project_dir, 'experiments.json')
        
        if os.path.exists(experiments_file):
            with open(experiments_file, 'r') as f:
                data = json.load(f)
                # Reconstruct experiment objects
                for exp_id, exp_data in data.items():
                    exp = Experiment(exp_data['name'])
                    exp.__dict__.update(exp_data)
                    experiments[exp_id] = exp
                    
        return experiments
        
    def create_experiment(self, 
                         name: str,
                         config: Optional[Dict[str, Any]] = None,
                         tags: Optional[List[str]] = None) -> str:
        """Create a new experiment."""
        if self.current_experiment and self.current_experiment.status == 'running':
            warnings.warn("Previous experiment still running. Ending it automatically.")
            self.end_experiment()
            
        experiment = Experiment(name, tags, config)
        self.current_experiment = experiment
        self.experiments[experiment.id] = experiment
        self.step = 0
        
        # Create experiment directory
        exp_dir = os.path.join(self.project_dir, experiment.id)
        os.makedirs(exp_dir, exist_ok=True)
        
        # Save initial state
        if self.auto_save:
            self._save_experiments()
            
        return experiment.id
        
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics for current step."""
        if not self.current_experiment:
            raise ValueError("No active experiment. Call create_experiment first.")
            
        if step is None:
            step = self.step
            self.step += 1
        else:
            self.step = step + 1
            
        # Record metrics
        for name, value in metrics.items():
            self.current_experiment.metrics[name].append({
                'step': step,
                'value': value,
                'timestamp': time.time()
            })
            
        # Auto-save if needed
        if self.auto_save and step % self.save_interval == 0:
            self._save_experiments()
            
    def log_hyperparameters(self, params: Dict[str, Any]):
        """Log hyperparameters."""
        if not self.current_experiment:
            raise ValueError("No active experiment.")
            
        self.current_experiment.config.update(params)
        
        if self.auto_save:
            self._save_experiments()
            
    def log_artifact(self, name: str, artifact: Any, metadata: Optional[Dict[str, Any]] = None):
        """Log an artifact (model, data, etc.)."""
        if not self.current_experiment:
            raise ValueError("No active experiment.")
            
        artifact_dir = os.path.join(self.project_dir, self.current_experiment.id, 'artifacts')
        os.makedirs(artifact_dir, exist_ok=True)
        
        # Determine artifact type and save
        artifact_path = os.path.join(artifact_dir, name)
        
        if isinstance(artifact, nn.Module):
            # Save model
            torch.save({
                'state_dict': artifact.state_dict(),
                'config': getattr(artifact, 'config', {}),
                'metadata': metadata or {}
            }, artifact_path + '.pt')
        elif isinstance(artifact, dict):
            # Save dictionary
            with open(artifact_path + '.json', 'w') as f:
                json.dump(artifact, f, indent=2)
        elif isinstance(artifact, (list, tuple)):
            # Save as pickle
            with open(artifact_path + '.pkl', 'wb') as f:
                pickle.dump(artifact, f)
        else:
            # Save as pickle for other types
            with open(artifact_path + '.pkl', 'wb') as f:
                pickle.dump(artifact, f)
                
        # Record artifact info
        self.current_experiment.artifacts[name] = {
            'path': artifact_path,
            'type': type(artifact).__name__,
            'metadata': metadata or {},
            'created_at': time.time()
        }
        
    def save_checkpoint(self, 
                       model: nn.Module,
                       optimizer: Optional[torch.optim.Optimizer] = None,
                       epoch: Optional[int] = None,
                       metrics: Optional[Dict[str, float]] = None):
        """Save model checkpoint."""
        if not self.current_experiment:
            raise ValueError("No active experiment.")
            
        checkpoint_dir = os.path.join(self.project_dir, self.current_experiment.id, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Generate checkpoint name
        checkpoint_name = f"checkpoint_step_{self.step}"
        if epoch is not None:
            checkpoint_name = f"checkpoint_epoch_{epoch}"
            
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name + '.pt')
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'step': self.step,
            'model_state_dict': model.state_dict(),
            'metrics': metrics or {}
        }
        
        if optimizer:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
            
        torch.save(checkpoint, checkpoint_path)
        
        # Record checkpoint
        self.current_experiment.checkpoints.append({
            'path': checkpoint_path,
            'epoch': epoch,
            'step': self.step,
            'metrics': metrics or {},
            'created_at': time.time()
        })
        
        # Version control if enabled
        if self.version_control:
            self._version_checkpoint(checkpoint_path)
            
    def load_checkpoint(self, 
                       checkpoint_id: Union[str, int],
                       model: nn.Module,
                       optimizer: Optional[torch.optim.Optimizer] = None) -> Dict[str, Any]:
        """Load a checkpoint."""
        if not self.current_experiment:
            raise ValueError("No active experiment.")
            
        # Find checkpoint
        if isinstance(checkpoint_id, int):
            # Load by index
            checkpoint_info = self.current_experiment.checkpoints[checkpoint_id]
        else:
            # Load by path
            checkpoint_info = None
            for ckpt in self.current_experiment.checkpoints:
                if checkpoint_id in ckpt['path']:
                    checkpoint_info = ckpt
                    break
                    
        if not checkpoint_info:
            raise ValueError(f"Checkpoint {checkpoint_id} not found.")
            
        # Load checkpoint
        checkpoint = torch.load(checkpoint_info['path'])
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        return checkpoint
        
    def compare_experiments(self, 
                          experiment_ids: List[str],
                          metrics: Optional[List[str]] = None) -> Dict[str, Any]:
        """Compare multiple experiments."""
        comparison = {
            'experiments': {},
            'metrics': {}
        }
        
        for exp_id in experiment_ids:
            if exp_id not in self.experiments:
                warnings.warn(f"Experiment {exp_id} not found.")
                continue
                
            exp = self.experiments[exp_id]
            comparison['experiments'][exp_id] = {
                'name': exp.name,
                'config': exp.config,
                'status': exp.status,
                'duration': exp.duration()
            }
            
        # Compare metrics
        metric_names = metrics
        if not metric_names:
            # Use all available metrics
            metric_names = set()
            for exp_id in experiment_ids:
                if exp_id in self.experiments:
                    metric_names.update(self.experiments[exp_id].metrics.keys())
                    
        for metric_name in metric_names:
            comparison['metrics'][metric_name] = {}
            for exp_id in experiment_ids:
                if exp_id in self.experiments:
                    exp = self.experiments[exp_id]
                    if metric_name in exp.metrics:
                        values = [m['value'] for m in exp.metrics[metric_name]]
                        comparison['metrics'][metric_name][exp_id] = {
                            'final': values[-1] if values else None,
                            'best': max(values) if values else None,
                            'mean': sum(values) / len(values) if values else None
                        }
                        
        return comparison
        
    def get_best_experiment(self, metric: str, mode: str = 'max') -> Optional[str]:
        """Get best experiment based on metric."""
        best_exp_id = None
        best_value = None
        
        for exp_id, exp in self.experiments.items():
            if metric in exp.metrics and exp.metrics[metric]:
                values = [m['value'] for m in exp.metrics[metric]]
                
                if mode == 'max':
                    value = max(values)
                    if best_value is None or value > best_value:
                        best_value = value
                        best_exp_id = exp_id
                else:  # min
                    value = min(values)
                    if best_value is None or value < best_value:
                        best_value = value
                        best_exp_id = exp_id
                        
        return best_exp_id
        
    def end_experiment(self):
        """End current experiment."""
        if not self.current_experiment:
            warnings.warn("No active experiment to end.")
            return
            
        self.current_experiment.status = 'completed'
        self.current_experiment.end_time = datetime.now()
        
        # Save final state
        self._save_experiments()
        
        # Generate summary
        summary = self._generate_experiment_summary(self.current_experiment)
        summary_path = os.path.join(
            self.project_dir, 
            self.current_experiment.id, 
            'summary.json'
        )
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
            
        self.current_experiment = None
        
    def _save_experiments(self):
        """Save experiments to disk."""
        experiments_data = {}
        for exp_id, exp in self.experiments.items():
            exp_dict = exp.__dict__.copy()
            # Convert datetime objects
            exp_dict['start_time'] = exp.start_time.isoformat()
            if exp.end_time:
                exp_dict['end_time'] = exp.end_time.isoformat()
            experiments_data[exp_id] = exp_dict
            
        experiments_file = os.path.join(self.project_dir, 'experiments.json')
        with open(experiments_file, 'w') as f:
            json.dump(experiments_data, f, indent=2)
            
    def _version_checkpoint(self, checkpoint_path: str):
        """Version control for checkpoints."""
        # Simple versioning by copying to versions directory
        versions_dir = os.path.join(
            os.path.dirname(checkpoint_path), 
            'versions'
        )
        os.makedirs(versions_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        version_path = os.path.join(
            versions_dir,
            f"{os.path.basename(checkpoint_path)}_{timestamp}"
        )
        shutil.copy2(checkpoint_path, version_path)
        
    def _generate_experiment_summary(self, experiment: Experiment) -> Dict[str, Any]:
        """Generate experiment summary."""
        summary = {
            'id': experiment.id,
            'name': experiment.name,
            'status': experiment.status,
            'duration_seconds': experiment.duration(),
            'config': experiment.config,
            'tags': experiment.tags,
            'metrics_summary': {}
        }
        
        # Summarize metrics
        for metric_name, values in experiment.metrics.items():
            if values:
                metric_values = [v['value'] for v in values]
                summary['metrics_summary'][metric_name] = {
                    'final': metric_values[-1],
                    'best': max(metric_values),
                    'worst': min(metric_values),
                    'mean': sum(metric_values) / len(metric_values),
                    'n_values': len(metric_values)
                }
                
        # Add artifacts info
        summary['artifacts'] = list(experiment.artifacts.keys())
        summary['n_checkpoints'] = len(experiment.checkpoints)
        
        return summary
        
    def visualize_metrics(self, 
                         metric_names: List[str],
                         experiment_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """Prepare metrics for visualization."""
        if experiment_ids is None:
            experiment_ids = list(self.experiments.keys())
            
        visualization_data = {
            'metrics': defaultdict(dict)
        }
        
        for exp_id in experiment_ids:
            if exp_id not in self.experiments:
                continue
                
            exp = self.experiments[exp_id]
            for metric_name in metric_names:
                if metric_name in exp.metrics:
                    steps = [m['step'] for m in exp.metrics[metric_name]]
                    values = [m['value'] for m in exp.metrics[metric_name]]
                    
                    visualization_data['metrics'][metric_name][exp_id] = {
                        'steps': steps,
                        'values': values,
                        'name': exp.name
                    }
                    
        return visualization_data
        
    def export_experiment(self, experiment_id: str, output_path: str):
        """Export entire experiment."""
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found.")
            
        exp_dir = os.path.join(self.project_dir, experiment_id)
        if os.path.exists(exp_dir):
            shutil.make_archive(output_path, 'zip', exp_dir)
        else:
            warnings.warn(f"Experiment directory {exp_dir} not found.")