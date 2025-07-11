#!/usr/bin/env python3
"""
Enhanced Training script for the Raga Classifier ML model with comprehensive improvements.
Includes data preprocessing, feature engineering, multiple models, hyperparameter tuning,
cross-validation, ensemble methods, and robust evaluation.
"""

import os
import yaml
import pickle
import warnings
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
)
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_fscore_support, roc_auc_score
)
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
from data_loader import load_audio_samples, dataframe_to_features_labels
import random

# Optional imports with fallbacks
try:
    from xgboost import XGBClassifier
    has_xgb = True
except ImportError:
    has_xgb = False
    print("XGBoost not available. Install with: pip install xgboost")

try:
    from category_encoders import TargetEncoder
    has_target_encoder = True
except ImportError:
    has_target_encoder = False
    print("Category encoders not available. Install with: pip install category-encoders")

try:
    import optuna
    from optuna.integration import OptunaSearchCV
    has_optuna = True
    print("Using Optuna for Bayesian hyperparameter optimization")
except ImportError:
    has_optuna = False
    print("Optuna not available. Install with: pip install optuna")
    print("Falling back to GridSearchCV")

try:
    from sklearn.model_selection import learning_curve
    has_learning_curve = True
except ImportError:
    has_learning_curve = False

# Configuration
CONFIG = {
    'random_state': 42,
    'test_size': 0.2,
    'cv_folds': 5,
    'n_jobs': -1,
    'min_samples_per_class': 5,
    'max_features': 1000,
    'smote_sampling_strategy': 'minority',
    'optuna_trials': 50,  # Number of Optuna trials
    'optuna_timeout': 300,  # Timeout in seconds
    'models': {
        'RandomForest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'class_weight': ['balanced']
        },
        'SVM': {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto'],
            'kernel': ['rbf', 'linear'],
            'class_weight': ['balanced']
        },
        'MLP': {
            'hidden_layer_sizes': [(64, 32), (128, 64, 32), (256, 128, 64)],
            'activation': ['relu', 'tanh'],
            'solver': ['adam'],
            'max_iter': [1000],
            'alpha': [0.0001, 0.001]
        }
    },
    # Optuna parameter distributions
    'optuna_distributions': {
        'RandomForest': {
            'n_estimators': ('int', 50, 200),
            'max_depth': ('int', 5, 30),
            'min_samples_split': ('int', 2, 20),
            'min_samples_leaf': ('int', 1, 10),
            'max_features': ('categorical', ['sqrt', 'log2', None])
        },
        'SVM': {
            'C': ('float', 0.01, 100.0),
            'gamma': ('categorical', ['scale', 'auto']),
            'kernel': ('categorical', ['rbf', 'linear', 'poly'])
        },
        'MLP': {
            'hidden_layer_sizes': ('categorical', [(64, 32), (128, 64, 32), (256, 128, 64), (512, 256, 128)]),
            'activation': ('categorical', ['relu', 'tanh']),
            'alpha': ('float', 0.0001, 0.01),
            'learning_rate_init': ('float', 0.001, 0.1)
        }
    }
}

if has_xgb:
    CONFIG['models']['XGBoost'] = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 6, 10],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    CONFIG['optuna_distributions']['XGBoost'] = {
        'n_estimators': ('int', 50, 300),
        'max_depth': ('int', 3, 15),
        'learning_rate': ('float', 0.01, 0.3),
        'subsample': ('float', 0.6, 1.0),
        'colsample_bytree': ('float', 0.6, 1.0),
        'reg_alpha': ('float', 0.0, 1.0),
        'reg_lambda': ('float', 0.0, 1.0)
    }

warnings.filterwarnings('ignore')

class RagaClassifierTrainer:
    """Enhanced trainer for raga classification with comprehensive ML pipeline."""
    
    def __init__(self, config=None):
        self.config = config or CONFIG
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.feature_selector = SelectKBest(f_classif, k=self.config['max_features'])
        self.smote = SMOTE(random_state=self.config['random_state'])
        self.best_models = {}
        self.ensemble_model = None
        
    def load_and_preprocess_data(self):
        """Load data and perform comprehensive preprocessing."""
        print("Loading data...")
        df = load_audio_samples()
        
        # Store original raga names
        self.original_raga_names = df['raga_name'].copy()
        
        # Define metadata features
        self.meta_features = [
            'type_name', 'audio_region', 'composer_name', 
            'tala_name', 'song_title', 'artist_name'
        ]
        
        print(f'Total samples loaded: {len(df)}')
        print(f'Unique raga names: {self.original_raga_names.nunique()}')
        
        # Check class distribution
        class_counts = self.original_raga_names.value_counts()
        print('Raga distribution:')
        print(class_counts)
        
        # Filter classes with minimum samples
        min_samples = self.config['min_samples_per_class']
        valid_classes = class_counts[class_counts >= min_samples].index
        df_filtered = df[df['raga_name'].isin(valid_classes)]
        
        print(f'Classes with >= {min_samples} samples: {len(valid_classes)}')
        print(f'Samples after filtering: {len(df_filtered)}')
        
        return df_filtered
    
    def encode_features(self, df):
        """Enhanced feature encoding with multiple strategies."""
        df_encoded = df.copy()
        
        if has_target_encoder:
            # Use target encoding for better categorical handling
            target_encoder = TargetEncoder()
            for col in self.meta_features:
                df_encoded[col] = target_encoder.fit_transform(
                    df_encoded[col], df_encoded['raga_name']
                )
        else:
            # Fallback to categorical codes
            for col in self.meta_features:
                df_encoded[col] = df_encoded[col].astype('category').cat.codes
        
        return df_encoded
    
    def extract_features_labels(self, df, augment=False):
        """Extract features and labels with proper encoding. Optionally augment audio."""
        # Extract features using existing function
        X, y_strings = dataframe_to_features_labels(
            df, self.meta_features, label_col='raga_name', augment=augment
        )
        # Encode labels
        y = self.label_encoder.fit_transform(y_strings)
        print(f'Feature shape: {X.shape}')
        print(f'Unique classes: {len(np.unique(y))}')
        print(f'Class distribution: {Counter(y)}')
        return X, y
    
    def prepare_data(self, X, y):
        """Comprehensive data preparation including scaling and feature selection."""
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config['test_size'], 
            random_state=self.config['random_state'], 
            stratify=y
        )
        
        # Feature scaling
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Feature selection
        if X_train_scaled.shape[1] > self.config['max_features']:
            X_train_selected = self.feature_selector.fit_transform(X_train_scaled, y_train)
            X_test_selected = self.feature_selector.transform(X_test_scaled)
        else:
            X_train_selected = X_train_scaled
            X_test_selected = X_test_scaled
        
        # Handle class imbalance with SMOTE
        X_train_balanced, y_train_balanced = self.smote.fit_resample(
            X_train_selected, y_train
        )
        
        print(f'Training set shape after preprocessing: {X_train_balanced.shape}')
        print(f'Class distribution after balancing: {Counter(y_train_balanced)}')
        
        return X_train_balanced, X_test_selected, y_train_balanced, y_test
    
    def create_optuna_objective(self, model_class, model_name, X_train, y_train):
        """Create Optuna objective function for hyperparameter optimization."""
        def objective(trial):
            # Get parameter distributions for this model
            param_distributions = self.config['optuna_distributions'][model_name]
            params = {}
            
            # Sample parameters based on distributions
            for param_name, (dist_type, *args) in param_distributions.items():
                if dist_type == 'int':
                    params[param_name] = trial.suggest_int(param_name, args[0], args[1])
                elif dist_type == 'float':
                    params[param_name] = trial.suggest_float(param_name, args[0], args[1])
                elif dist_type == 'categorical':
                    params[param_name] = trial.suggest_categorical(param_name, args[0])
            
            # Add model-specific parameters
            if model_name == 'RandomForest':
                params['class_weight'] = 'balanced'
            elif model_name == 'SVM':
                params['class_weight'] = 'balanced'
                params['probability'] = True
            elif model_name == 'MLP':
                params['max_iter'] = 1000
                params['solver'] = 'adam'
            elif model_name == 'XGBoost':
                params['use_label_encoder'] = False
                params['eval_metric'] = 'mlogloss'
            
            # Set random state
            params['random_state'] = self.config['random_state']
            
            # Create and train model
            model = model_class(**params)
            
            # Cross-validation
            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=StratifiedKFold(n_splits=self.config['cv_folds'], shuffle=True,
                                  random_state=self.config['random_state']),
                scoring='accuracy',
                n_jobs=1  # Use 1 job to avoid conflicts with Optuna parallelization
            )
            
            return cv_scores.mean()
        
        return objective
        
    def train_single_model(self, model_name, model_class, param_grid, X_train, y_train):
        """Train a single model with Optuna Bayesian optimization or GridSearchCV fallback."""
        print(f'\n--- Training {model_name} ---')
        
        if has_optuna and model_name in self.config['optuna_distributions']:
            print(f'Using Optuna Bayesian optimization for {model_name}')
            
            # Create Optuna study
            study = optuna.create_study(
                direction='maximize',
                sampler=optuna.samplers.TPESampler(seed=self.config['random_state'])
            )
            
            # Create objective function
            objective = self.create_optuna_objective(model_class, model_name, X_train, y_train)
            
            # Optimize
            study.optimize(
                objective,
                n_trials=self.config['optuna_trials'],
                timeout=self.config['optuna_timeout'],
                n_jobs=1,  # Use 1 job to avoid conflicts
                show_progress_bar=True
            )
            
            print(f'Best parameters: {study.best_params}')
            print(f'Best CV score: {study.best_value:.4f}')
            print(f'Number of trials: {len(study.trials)}')
            
            # Train final model with best parameters
            best_params = study.best_params.copy()
            
            # Add model-specific parameters
            if model_name == 'RandomForest':
                best_params['class_weight'] = 'balanced'
            elif model_name == 'SVM':
                best_params['class_weight'] = 'balanced'
                best_params['probability'] = True
            elif model_name == 'MLP':
                best_params['max_iter'] = 1000
                best_params['solver'] = 'adam'
            elif model_name == 'XGBoost':
                best_params['use_label_encoder'] = False
                best_params['eval_metric'] = 'mlogloss'
            
            best_params['random_state'] = self.config['random_state']
            best_model = model_class(**best_params)
            best_model.fit(X_train, y_train)
            
            # Store optimization results for logging
            best_model._optuna_study = study
            
        else:
            print(f'Using GridSearchCV for {model_name}')
            
            # Create base model
            if model_name == 'XGBoost' and has_xgb:
                base_model = model_class(
                    use_label_encoder=False,
                    eval_metric='mlogloss',
                    random_state=self.config['random_state']
                )
            else:
                base_model = model_class(random_state=self.config['random_state'])
            
            # Grid search with cross-validation
            grid_search = GridSearchCV(
                base_model, param_grid,
                cv=StratifiedKFold(n_splits=self.config['cv_folds'], shuffle=True,
                                  random_state=self.config['random_state']),
                scoring='accuracy',
                n_jobs=self.config['n_jobs'],
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            
            print(f'Best parameters: {grid_search.best_params_}')
            print(f'Best CV score: {grid_search.best_score_:.4f}')
            
            best_model = grid_search.best_estimator_
            best_model._grid_search = grid_search
        
        return best_model
    
    def evaluate_model(self, model, model_name, X_test, y_test):
        """Comprehensive model evaluation."""
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted'
        )
        
        # AUC for multiclass
        auc = None
        if y_pred_proba is not None and len(np.unique(y_test)) > 2:
            try:
                auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
            except ValueError:
                pass
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc
        }
        
        print(f'[{model_name}] Results:')
        for metric, value in results.items():
            if value is not None:
                print(f'  {metric}: {value:.4f}')
        
        return results, y_pred
    
    def plot_learning_curves(self, model, model_name, X_train, y_train):
        """Generate learning curves for the model."""
        if not has_learning_curve:
            return None
            
        print(f'Generating learning curves for {model_name}...')
        
        # Calculate learning curves
        train_sizes = np.linspace(0.1, 1.0, 10)
        train_sizes_abs, train_scores, val_scores = learning_curve(
            model, X_train, y_train,
            train_sizes=train_sizes,
            cv=StratifiedKFold(n_splits=self.config['cv_folds'], shuffle=True,
                              random_state=self.config['random_state']),
            scoring='accuracy',
            n_jobs=self.config['n_jobs']
        )
        
        # Calculate mean and std
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes_abs, train_mean, 'o-', color='blue', label='Training accuracy')
        plt.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, 
                        alpha=0.1, color='blue')
        
        plt.plot(train_sizes_abs, val_mean, 'o-', color='red', label='Validation accuracy')
        plt.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std, 
                        alpha=0.1, color='red')
        
        plt.xlabel('Training Set Size')
        plt.ylabel('Accuracy')
        plt.title(f'Learning Curves: {model_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        learning_curve_path = f'learning_curves_{model_name}.png'
        plt.savefig(learning_curve_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return learning_curve_path, {
            'train_sizes': train_sizes_abs.tolist(),
            'train_scores_mean': train_mean.tolist(),
            'train_scores_std': train_std.tolist(),
            'val_scores_mean': val_mean.tolist(),
            'val_scores_std': val_std.tolist()
        }
    
    def plot_feature_importance(self, model, model_name, feature_names=None):
        """Generate feature importance plot."""
        if not hasattr(model, 'feature_importances_'):
            return None
            
        print(f'Generating feature importance plot for {model_name}...')
        
        importances = model.feature_importances_
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(importances))]
        
        # Get top 20 features
        indices = np.argsort(importances)[::-1][:20]
        
        plt.figure(figsize=(12, 8))
        plt.title(f'Feature Importance: {model_name}')
        plt.bar(range(len(indices)), importances[indices])
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45)
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.tight_layout()
        
        feature_importance_path = f'feature_importance_{model_name}.png'
        plt.savefig(feature_importance_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create importance dataframe
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return feature_importance_path, importance_df
    
    def plot_optuna_optimization(self, study, model_name):
        """Generate Optuna optimization plots."""
        try:
            import optuna.visualization as vis
            
            # Optimization history
            fig1 = vis.plot_optimization_history(study)
            fig1.write_image(f'optuna_history_{model_name}.png')
            
            # Parameter importance
            fig2 = vis.plot_param_importances(study)
            fig2.write_image(f'optuna_param_importance_{model_name}.png')
            
            # Parallel coordinate plot
            fig3 = vis.plot_parallel_coordinate(study)
            fig3.write_image(f'optuna_parallel_coordinate_{model_name}.png')
            
            return [f'optuna_history_{model_name}.png', 
                   f'optuna_param_importance_{model_name}.png',
                   f'optuna_parallel_coordinate_{model_name}.png']
            
        except ImportError:
            print("Plotly not available for Optuna visualization")
            return []
        except Exception as e:
            print(f"Error generating Optuna plots: {e}")
            return []
    
    def log_to_mlflow(self, model, model_name, results, y_test, y_pred, X_train, y_train):
        """Enhanced MLflow logging with learning curves and feature importance."""
        with mlflow.start_run(run_name=f"{model_name}_enhanced"):
            # Log parameters
            mlflow.log_param('model', model_name)
            mlflow.log_param('features', self.meta_features + ['audio_features'])
            mlflow.log_param('num_classes', len(self.label_encoder.classes_))
            mlflow.log_param('num_samples', len(X_train))
            mlflow.log_param('feature_selection', self.config['max_features'])
            mlflow.log_param('smote_used', True)
            
            # Log optimization method
            if hasattr(model, '_optuna_study'):
                mlflow.log_param('optimization_method', 'Optuna')
                mlflow.log_param('optuna_trials', len(model._optuna_study.trials))
                mlflow.log_param('optuna_best_value', model._optuna_study.best_value)
            elif hasattr(model, '_grid_search'):
                mlflow.log_param('optimization_method', 'GridSearchCV')
                mlflow.log_param('grid_search_best_score', model._grid_search.best_score_)
            
            # Log metrics
            for metric, value in results.items():
                if value is not None:
                    mlflow.log_metric(metric, value)
            
            # Cross-validation scores
            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=StratifiedKFold(n_splits=self.config['cv_folds'], shuffle=True,
                                  random_state=self.config['random_state'])
            )
            mlflow.log_metric('cv_mean_accuracy', cv_scores.mean())
            mlflow.log_metric('cv_std_accuracy', cv_scores.std())
            
            # Log model
            mlflow.sklearn.log_model(model, 'model')
            
            # Log preprocessing objects
            with open('label_encoder.pkl', 'wb') as f:
                pickle.dump(self.label_encoder, f)
            with open('scaler.pkl', 'wb') as f:
                pickle.dump(self.scaler, f)
            with open('feature_selector.pkl', 'wb') as f:
                pickle.dump(self.feature_selector, f)
            
            mlflow.log_artifact('label_encoder.pkl')
            mlflow.log_artifact('scaler.pkl')
            mlflow.log_artifact('feature_selector.pkl')
            
            # Generate and log learning curves
            learning_curve_result = self.plot_learning_curves(model, model_name, X_train, y_train)
            if learning_curve_result:
                learning_curve_path, learning_curve_data = learning_curve_result
                mlflow.log_artifact(learning_curve_path)
                mlflow.log_dict(learning_curve_data, 'learning_curve_data.json')
            
            # Generate and log feature importance
            feature_importance_result = self.plot_feature_importance(model, model_name)
            if feature_importance_result:
                feature_importance_path, importance_df = feature_importance_result
                mlflow.log_artifact(feature_importance_path)
                mlflow.log_text(importance_df.to_string(), 'feature_importance.txt')
            
            # Log Optuna optimization plots
            if hasattr(model, '_optuna_study'):
                optuna_plots = self.plot_optuna_optimization(model._optuna_study, model_name)
                for plot_path in optuna_plots:
                    if os.path.exists(plot_path):
                        mlflow.log_artifact(plot_path)
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(12, 10))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=self.label_encoder.classes_,
                       yticklabels=self.label_encoder.classes_)
            plt.title(f'Confusion Matrix: {model_name}')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
            plt.tight_layout()
            cm_path = f'confusion_matrix_{model_name}.png'
            plt.savefig(cm_path, dpi=300, bbox_inches='tight')
            mlflow.log_artifact(cm_path)
            plt.close()
            
            # Classification report
            report = classification_report(y_test, y_pred, 
                                         target_names=self.label_encoder.classes_)
            mlflow.log_text(report, 'classification_report.txt')
            
            # Misclassified samples analysis
            misclassified_mask = y_test != y_pred
            if np.any(misclassified_mask):
                misclassified_df = pd.DataFrame({
                    'true_label': self.label_encoder.inverse_transform(y_test[misclassified_mask]),
                    'predicted_label': self.label_encoder.inverse_transform(y_pred[misclassified_mask])
                })
                mlflow.log_text(misclassified_df.to_string(), 'misclassified_samples.txt')
            
            # Clean up temporary files
            temp_files = [
                'label_encoder.pkl', 'scaler.pkl', 'feature_selector.pkl', 
                cm_path, f'learning_curves_{model_name}.png', 
                f'feature_importance_{model_name}.png'
            ]
            
            # Add Optuna plot files to cleanup
            if hasattr(model, '_optuna_study'):
                temp_files.extend([
                    f'optuna_history_{model_name}.png',
                    f'optuna_param_importance_{model_name}.png',
                    f'optuna_parallel_coordinate_{model_name}.png'
                ])
            
            for file in temp_files:
                if os.path.exists(file):
                    os.remove(file)
    
    def create_ensemble_models(self, models, X_train, y_train):
        """Create ensemble models from trained individual models."""
        print("\n--- Creating Ensemble Models ---")
        
        # Voting Classifier
        voting_estimators = [(name, model) for name, model in models.items()]
        voting_model = VotingClassifier(
            estimators=voting_estimators,
            voting='soft'
        )
        voting_model.fit(X_train, y_train)
        
        # Stacking Classifier
        stacking_model = StackingClassifier(
            estimators=voting_estimators,
            final_estimator=MLPClassifier(
                hidden_layer_sizes=(64, 32),
                random_state=self.config['random_state'],
                max_iter=1000
            )
        )
        stacking_model.fit(X_train, y_train)
        
        return {'Voting': voting_model, 'Stacking': stacking_model}
    
    def train_all_models(self):
        """Main training pipeline."""
        print("=== Enhanced Raga Classifier Training ===")
        
        # Load and preprocess data
        df = self.load_and_preprocess_data()
        df_encoded = self.encode_features(df)
        
        # Split DataFrame into train and test
        train_df, test_df = train_test_split(
            df_encoded, test_size=self.config['test_size'], random_state=self.config['random_state'], stratify=df_encoded['raga_name']
        )
        
        # Extract features and labels (augment only for training)
        X_train, y_train = self.extract_features_labels(train_df, augment=True)
        X_test, y_test = self.extract_features_labels(test_df, augment=False)
        
        # Prepare data (scaling, feature selection, SMOTE)
        X_train, X_test, y_train, y_test = self.prepare_data(X_train, y_train)
        
        # Set up MLflow
        mlflow.set_experiment('enhanced_raga_classification')
        
        # Train individual models
        model_classes = {
            'RandomForest': RandomForestClassifier,
            'SVM': SVC,
            'MLP': MLPClassifier
        }
        
        if has_xgb:
            model_classes['XGBoost'] = XGBClassifier
        
        trained_models = {}
        all_results = {}
        
        for model_name, model_class in model_classes.items():
            if model_name in self.config['models']:
                param_grid = self.config['models'][model_name]
                
                # Train model
                best_model = self.train_single_model(
                    model_name, model_class, param_grid, X_train, y_train
                )
                
                # Evaluate model
                results, y_pred = self.evaluate_model(
                    best_model, model_name, X_test, y_test
                )
                
                # Log to MLflow
                self.log_to_mlflow(
                    best_model, model_name, results, y_test, y_pred, X_train, y_train
                )
                
                trained_models[model_name] = best_model
                all_results[model_name] = results
        
        # Create and evaluate ensemble models
        if len(trained_models) > 1:
            ensemble_models = self.create_ensemble_models(trained_models, X_train, y_train)
            
            for ensemble_name, ensemble_model in ensemble_models.items():
                results, y_pred = self.evaluate_model(
                    ensemble_model, ensemble_name, X_test, y_test
                )
                
                self.log_to_mlflow(
                    ensemble_model, ensemble_name, results, y_test, y_pred, X_train, y_train
                )
                
                all_results[ensemble_name] = results
        
        # Print final summary
        print("\n=== Final Results Summary ===")
        results_df = pd.DataFrame(all_results).T
        print(results_df.round(4))
        
        # Find best model
        best_model_name = results_df['accuracy'].idxmax()
        print(f"\nBest model: {best_model_name} (Accuracy: {results_df.loc[best_model_name, 'accuracy']:.4f})")
        
        return trained_models, all_results

def main():
    """Main execution function."""
    trainer = RagaClassifierTrainer()
    
    try:
        trained_models, results = trainer.train_all_models()
        print("\n=== Training Complete ===")
        print("All models trained and logged to MLflow.")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    main()