import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
import numpy as np
from dataclasses import dataclass
import json
import time
from collections import defaultdict
import warnings


@dataclass
class ParameterSpace:
    """Define a hyperparameter search space."""
    name: str
    type: str  # 'int', 'float', 'categorical', 'bool'
    bounds: Optional[Tuple[float, float]] = None
    choices: Optional[List[Any]] = None
    log_scale: bool = False
    
    def sample(self, rng: np.random.RandomState) -> Any:
        """Sample a value from this parameter space."""
        if self.type == 'int':
            if self.log_scale:
                return int(np.exp(rng.uniform(np.log(self.bounds[0]), np.log(self.bounds[1]))))
            return rng.randint(self.bounds[0], self.bounds[1] + 1)
        elif self.type == 'float':
            if self.log_scale:
                return np.exp(rng.uniform(np.log(self.bounds[0]), np.log(self.bounds[1])))
            return rng.uniform(self.bounds[0], self.bounds[1])
        elif self.type == 'categorical':
            return rng.choice(self.choices)
        elif self.type == 'bool':
            return rng.choice([True, False])
        else:
            raise ValueError(f"Unknown parameter type: {self.type}")


@dataclass
class Trial:
    """Represents a single optimization trial."""
    id: int
    parameters: Dict[str, Any]
    objective: Optional[float] = None
    constraints: Dict[str, float] = None
    metadata: Dict[str, Any] = None
    status: str = 'pending'  # 'pending', 'running', 'completed', 'failed'
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    
    def duration(self) -> Optional[float]:
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None


class AcquisitionFunction:
    """Base class for acquisition functions."""
    def __init__(self, exploration_weight: float = 1.0):
        self.exploration_weight = exploration_weight
        
    def __call__(self, mean: np.ndarray, std: np.ndarray, best_value: float) -> np.ndarray:
        raise NotImplementedError


class ExpectedImprovement(AcquisitionFunction):
    """Expected Improvement acquisition function."""
    def __call__(self, mean: np.ndarray, std: np.ndarray, best_value: float) -> np.ndarray:
        from scipy.stats import norm
        improvement = mean - best_value
        z = improvement / (std + 1e-9)
        ei = improvement * norm.cdf(z) + std * norm.pdf(z)
        return ei * self.exploration_weight


class UpperConfidenceBound(AcquisitionFunction):
    """Upper Confidence Bound acquisition function."""
    def __call__(self, mean: np.ndarray, std: np.ndarray, best_value: float) -> np.ndarray:
        return mean + self.exploration_weight * std


class HyperparameterOptimizer(nn.Module):
    def __init__(self,
                 parameter_space: List[ParameterSpace],
                 objective: str = 'maximize',  # 'maximize' or 'minimize'
                 n_trials: int = 100,
                 n_random_starts: int = 10,
                 acquisition: str = 'ei',  # 'ei' or 'ucb'
                 exploration_weight: float = 1.0,
                 constraints: Optional[List[Callable]] = None,
                 early_stopping_patience: int = 20,
                 seed: Optional[int] = None):
        super().__init__()
        self.parameter_space = {p.name: p for p in parameter_space}
        self.objective = objective
        self.n_trials = n_trials
        self.n_random_starts = n_random_starts
        self.acquisition = acquisition
        self.exploration_weight = exploration_weight
        self.constraints = constraints or []
        self.early_stopping_patience = early_stopping_patience
        self.rng = np.random.RandomState(seed)
        
        # Optimization state
        self.trials = []
        self.best_trial = None
        self.iteration = 0
        self.surrogate_model = None
        
        # Acquisition function
        if acquisition == 'ei':
            self.acquisition_fn = ExpectedImprovement(exploration_weight)
        elif acquisition == 'ucb':
            self.acquisition_fn = UpperConfidenceBound(exploration_weight)
        else:
            raise ValueError(f"Unknown acquisition function: {acquisition}")
            
    def suggest(self) -> Dict[str, Any]:
        """Suggest next hyperparameters to try."""
        if len(self.trials) < self.n_random_starts:
            # Random sampling for initial trials
            return self._random_sample()
        else:
            # Use Bayesian optimization
            return self._bayesian_sample()
            
    def _random_sample(self) -> Dict[str, Any]:
        """Random sampling from parameter space."""
        params = {}
        for name, space in self.parameter_space.items():
            params[name] = space.sample(self.rng)
        return params
        
    def _bayesian_sample(self) -> Dict[str, Any]:
        """Sample using Bayesian optimization."""
        # Train surrogate model on completed trials
        self._update_surrogate()
        
        # Generate candidate points
        n_candidates = 1000
        candidates = []
        for _ in range(n_candidates):
            candidate = self._random_sample()
            candidates.append(candidate)
            
        # Evaluate acquisition function
        acq_values = self._evaluate_acquisition(candidates)
        
        # Select best candidate
        best_idx = np.argmax(acq_values) if self.objective == 'maximize' else np.argmin(acq_values)
        return candidates[best_idx]
        
    def _update_surrogate(self):
        """Update surrogate model with completed trials."""
        # Simple Gaussian Process implementation
        X = []
        y = []
        
        for trial in self.trials:
            if trial.status == 'completed' and trial.objective is not None:
                x = self._params_to_vector(trial.parameters)
                X.append(x)
                y.append(trial.objective)
                
        if len(X) > 0:
            X = np.array(X)
            y = np.array(y)
            
            # Normalize objectives
            self.y_mean = np.mean(y)
            self.y_std = np.std(y) if np.std(y) > 0 else 1.0
            y_normalized = (y - self.y_mean) / self.y_std
            
            # Store for later use
            self.X_train = X
            self.y_train = y_normalized
            
    def _params_to_vector(self, params: Dict[str, Any]) -> np.ndarray:
        """Convert parameters to numeric vector."""
        vector = []
        for name, value in sorted(params.items()):
            space = self.parameter_space[name]
            if space.type in ['int', 'float']:
                if space.log_scale and value > 0:
                    vector.append(np.log(value))
                else:
                    vector.append(value)
            elif space.type == 'categorical':
                # One-hot encoding
                for choice in space.choices:
                    vector.append(1.0 if value == choice else 0.0)
            elif space.type == 'bool':
                vector.append(1.0 if value else 0.0)
        return np.array(vector)
        
    def _evaluate_acquisition(self, candidates: List[Dict[str, Any]]) -> np.ndarray:
        """Evaluate acquisition function for candidates."""
        if not hasattr(self, 'X_train'):
            return self.rng.rand(len(candidates))
            
        # Convert candidates to vectors
        X_candidates = np.array([self._params_to_vector(c) for c in candidates])
        
        # Simple distance-based prediction (placeholder for GP)
        means = []
        stds = []
        
        for x in X_candidates:
            # Compute distances to training points
            distances = np.linalg.norm(self.X_train - x, axis=1)
            weights = np.exp(-distances)
            weights /= np.sum(weights)
            
            # Weighted average for mean
            mean = np.sum(weights * self.y_train)
            # Uncertainty based on distance
            std = np.min(distances) + 0.1
            
            means.append(mean)
            stds.append(std)
            
        means = np.array(means) * self.y_std + self.y_mean
        stds = np.array(stds) * self.y_std
        
        # Get current best
        best_value = self.best_trial.objective if self.best_trial else 0.0
        
        # Compute acquisition values
        acq_values = self.acquisition_fn(means, stds, best_value)
        
        return acq_values
        
    def update(self, parameters: Dict[str, Any], objective: float, 
               constraints: Optional[Dict[str, float]] = None,
               metadata: Optional[Dict[str, Any]] = None):
        """Update optimizer with trial results."""
        # Find matching trial or create new one
        trial = None
        for t in self.trials:
            if t.parameters == parameters and t.status == 'running':
                trial = t
                break
                
        if trial is None:
            trial = Trial(
                id=len(self.trials),
                parameters=parameters,
                start_time=time.time()
            )
            self.trials.append(trial)
            
        # Update trial
        trial.objective = objective
        trial.constraints = constraints
        trial.metadata = metadata
        trial.status = 'completed'
        trial.end_time = time.time()
        
        # Update best trial
        if self._is_better(trial):
            self.best_trial = trial
            
        self.iteration += 1
        
    def _is_better(self, trial: Trial) -> bool:
        """Check if trial is better than current best."""
        if trial.objective is None:
            return False
            
        # Check constraints
        if trial.constraints:
            for name, value in trial.constraints.items():
                if value < 0:  # Constraint violated
                    return False
                    
        if self.best_trial is None:
            return True
            
        if self.objective == 'maximize':
            return trial.objective > self.best_trial.objective
        else:
            return trial.objective < self.best_trial.objective
            
    def should_stop(self) -> bool:
        """Check if optimization should stop early."""
        if self.iteration >= self.n_trials:
            return True
            
        # Check for convergence
        if len(self.trials) > self.early_stopping_patience:
            recent_trials = self.trials[-self.early_stopping_patience:]
            recent_objectives = [t.objective for t in recent_trials if t.objective is not None]
            
            if len(recent_objectives) > 0:
                # No improvement in recent trials
                best_recent = max(recent_objectives) if self.objective == 'maximize' else min(recent_objectives)
                if self.best_trial and abs(best_recent - self.best_trial.objective) < 1e-6:
                    return True
                    
        return False
        
    def get_best_parameters(self) -> Optional[Dict[str, Any]]:
        """Get best parameters found so far."""
        if self.best_trial:
            return self.best_trial.parameters
        return None
        
    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """Get full optimization history."""
        history = []
        for trial in self.trials:
            history.append({
                'id': trial.id,
                'parameters': trial.parameters,
                'objective': trial.objective,
                'constraints': trial.constraints,
                'status': trial.status,
                'duration': trial.duration()
            })
        return history
        
    def save_results(self, path: str):
        """Save optimization results to file."""
        results = {
            'best_parameters': self.get_best_parameters(),
            'best_objective': self.best_trial.objective if self.best_trial else None,
            'history': self.get_optimization_history(),
            'parameter_space': {name: {
                'type': space.type,
                'bounds': space.bounds,
                'choices': space.choices,
                'log_scale': space.log_scale
            } for name, space in self.parameter_space.items()},
            'config': {
                'objective': self.objective,
                'n_trials': self.n_trials,
                'acquisition': self.acquisition,
                'exploration_weight': self.exploration_weight
            }
        }
        
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)
            
    def forward(self, model_fn: Callable, eval_fn: Callable, 
                train_data: Any, val_data: Any) -> Dict[str, Any]:
        """Run full optimization loop."""
        while not self.should_stop():
            # Get next parameters to try
            params = self.suggest()
            
            # Create and train model
            try:
                model = model_fn(**params)
                model = self._train_model(model, train_data)
                
                # Evaluate
                objective = eval_fn(model, val_data)
                
                # Check constraints
                constraints = {}
                for i, constraint_fn in enumerate(self.constraints):
                    constraints[f'constraint_{i}'] = constraint_fn(model, params)
                    
                # Update optimizer
                self.update(params, objective, constraints)
                
            except Exception as e:
                # Mark trial as failed
                trial = Trial(
                    id=len(self.trials),
                    parameters=params,
                    status='failed',
                    metadata={'error': str(e)}
                )
                self.trials.append(trial)
                
        return {
            'best_parameters': self.get_best_parameters(),
            'best_objective': self.best_trial.objective if self.best_trial else None,
            'n_trials_completed': sum(1 for t in self.trials if t.status == 'completed')
        }
        
    def _train_model(self, model: nn.Module, train_data: Any) -> nn.Module:
        """Placeholder for model training."""
        # This would be implemented based on specific use case
        return model