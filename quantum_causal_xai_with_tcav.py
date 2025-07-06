import numpy as np
import pandas as pd
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import warnings
import os
from scipy.stats import pearsonr
from sklearn.feature_selection import mutual_info_regression

warnings.filterwarnings('ignore')

# Quantum-inspired components
from scipy.linalg import expm
from sklearn.kernel_approximation import RBFSampler

class QuantumInspiredKernel:
    """
    Quantum-inspired kernel using random Fourier features to approximate quantum feature maps.
    Optimized for performance with reduced components.
    """
    def __init__(self, n_components: int = 20, gamma: float = 1.0, quantum_depth: int = 2):
        self.n_components = n_components
        self.gamma = gamma
        self.quantum_depth = quantum_depth
        self.fourier_transformer = None
        self.quantum_params = None
        
    def _simulate_quantum_circuit(self, X: np.ndarray) -> np.ndarray:
        """Simulate a parameterized quantum circuit classically for feature embedding."""
        n_features = X.shape[1]
        
        if self.quantum_params is None:
            self.quantum_params = np.random.uniform(0, 2*np.pi, (self.quantum_depth, n_features))
        
        embedded_features = X.copy()
        
        for depth in range(self.quantum_depth):
            for i in range(n_features):
                embedded_features[:, i] = np.cos(self.quantum_params[depth, i] * embedded_features[:, i])
            if n_features > 1:
                for i in range(n_features - 1):
                    control = embedded_features[:, i]
                    target = embedded_features[:, i + 1]
                    embedded_features[:, i + 1] = np.sin(control * target)
        
        return embedded_features
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform features using quantum-inspired kernel."""
        quantum_features = self._simulate_quantum_circuit(X)
        self.fourier_transformer = RBFSampler(n_components=self.n_components, gamma=self.gamma, random_state=42)
        classical_features = self.fourier_transformer.fit_transform(X)
        return np.hstack([quantum_features, classical_features])
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform new data using fitted kernel."""
        if self.fourier_transformer is None:
            raise ValueError("Kernel not fitted. Call fit_transform first.")
        quantum_features = self._simulate_quantum_circuit(X)
        classical_features = self.fourier_transformer.transform(X)
        return np.hstack([quantum_features, classical_features])

class SimpleCausalDiscovery:
    """
    Simple causal discovery using correlation and mutual information.
    Replaces causal-learn dependency.
    """
    
    def __init__(self, correlation_threshold: float = 0.3, mi_threshold: float = 0.1):
        self.correlation_threshold = correlation_threshold
        self.mi_threshold = mi_threshold
        self.causal_graph = None
        self.feature_names = None
        self.correlation_matrix = None
        self.mutual_info_matrix = None
    
    def discover_causal_structure(self, X: np.ndarray, feature_names: List[str]) -> nx.DiGraph:
        """Discover causal structure using correlation and mutual information."""
        self.feature_names = feature_names
        n_features = X.shape[1]
        
        # Limit to original features only to avoid complexity
        n_original = min(len(feature_names), n_features)
        X_original = X[:, :n_original]
        
        # Compute correlation matrix
        self.correlation_matrix = np.corrcoef(X_original.T)
        
        # Compute mutual information matrix
        self.mutual_info_matrix = np.zeros((n_original, n_original))
        for i in range(n_original):
            for j in range(n_original):
                if i != j:
                    try:
                        mi = mutual_info_regression(X_original[:, [i]], X_original[:, j])
                        self.mutual_info_matrix[i, j] = mi[0]
                    except:
                        self.mutual_info_matrix[i, j] = 0
        
        # Create causal graph based on strong correlations and mutual information
        self.causal_graph = nx.DiGraph()
        self.causal_graph.add_nodes_from(range(n_original))
        
        for i in range(n_original):
            for j in range(n_original):
                if i != j:
                    corr = abs(self.correlation_matrix[i, j])
                    mi = self.mutual_info_matrix[i, j]
                    
                    # Add edge if both correlation and mutual information are above threshold
                    if corr > self.correlation_threshold and mi > self.mi_threshold:
                        self.causal_graph.add_edge(i, j)
        
        print(f"Discovered {self.causal_graph.number_of_edges()} causal relationships")
        return self.causal_graph
    
    def visualize_causal_graph(self, save_path: Optional[str] = None):
        """Visualize the discovered causal graph."""
        if not self.causal_graph or self.causal_graph.number_of_nodes() == 0:
            print("No causal graph to visualize.")
            return
        
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(self.causal_graph, k=2, iterations=50)
        
        # Draw nodes
        nx.draw_networkx_nodes(self.causal_graph, pos, node_color='lightblue', 
                              node_size=1500, alpha=0.7)
        
        # Draw edges
        nx.draw_networkx_edges(self.causal_graph, pos, edge_color='gray', 
                              arrows=True, arrowsize=20, alpha=0.6)
        
        # Draw labels
        labels = {i: name[:10] + "..." if len(name) > 10 else name 
                 for i, name in enumerate(self.feature_names[:self.causal_graph.number_of_nodes()])}
        nx.draw_networkx_labels(self.causal_graph, pos, labels, font_size=8)
        
        plt.title("Discovered Causal Relationships for Counterparty Credit Risk")
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

class CounterpartyRiskModel:
    """Main model combining quantum-inspired kernels, simple causal discovery, and TCAV explainability."""
    
    def __init__(self, quantum_components: int = 20, n_original_features: int = 6):
        self.quantum_kernel = QuantumInspiredKernel(n_components=quantum_components)
        self.causal_discovery = SimpleCausalDiscovery()
        self.risk_classifier = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
        self.scaler = StandardScaler()
        self.feature_names = None
        self.concepts = None
        self.n_original_features = n_original_features
        self.original_data = None
        
    def prepare_features(self, X: np.ndarray, feature_names: List[str]) -> np.ndarray:
        """Prepare features using quantum-inspired embedding."""
        self.feature_names = feature_names
        self.original_data = X.copy()
        X_scaled = self.scaler.fit_transform(X)
        return self.quantum_kernel.fit_transform(X_scaled)
    
    def discover_causal_relationships(self, X_quantum: np.ndarray):
        """Discover causal relationships using simple correlation-based method."""
        self.causal_graph = self.causal_discovery.discover_causal_structure(X_quantum, self.feature_names)
        return self.causal_graph
    
    def train_risk_model(self, X_quantum: np.ndarray, y: np.ndarray):
        """Train the counterparty risk classification model with SMOTE."""
        X_train, X_test, y_train, y_test = train_test_split(X_quantum, y, test_size=0.2, random_state=42)
        
        # Apply SMOTE to balance classes
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        
        self.risk_classifier.fit(X_train_balanced, y_train_balanced)
        y_pred = self.risk_classifier.predict(X_test)
        y_prob = self.risk_classifier.predict_proba(X_test)[:, 1]
        
        print("Model Performance:")
        print(f"AUC-ROC: {roc_auc_score(y_test, y_prob):.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return X_train_balanced, X_test, y_train_balanced, y_test
    
    def setup_concepts_for_tcav(self, X_quantum: np.ndarray, original_features: np.ndarray):
        """Define financial concepts for TCAV analysis."""
        self.concepts = {
            'liquidity': [0],  # AMT_INCOME_TOTAL
            'credit_exposure': [1, 2],  # AMT_CREDIT, AMT_ANNUITY
            'credit_score': [3, 4, 5],  # EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3
        }
    
    def compute_tcav_concept_importance(self, X_test: np.ndarray, n_perturbations: int = 100) -> Dict:
        """Compute TCAV-based concept importance for original features."""
        concept_importance = {}
        n_features = X_test.shape[1]
        
        for concept, indices in self.concepts.items():
            # Filter valid indices
            valid_indices = [i for i in indices if i < n_features]
            if not valid_indices:
                concept_importance[concept] = 0.0
                continue
            
            # Compute baseline predictions
            baseline_probs = self.risk_classifier.predict_proba(X_test)[:, 1]
            
            # Perturb concept features
            tcav_scores = []
            for _ in range(n_perturbations):
                X_perturbed = X_test.copy()
                # Apply small random perturbations to concept features
                perturbation = np.random.normal(0, 0.1, size=(X_test.shape[0], len(valid_indices)))
                X_perturbed[:, valid_indices] += perturbation
                
                # Compute change in predictions
                perturbed_probs = self.risk_classifier.predict_proba(X_perturbed)[:, 1]
                sensitivity = np.mean(np.abs(perturbed_probs - baseline_probs))
                tcav_scores.append(sensitivity)
            
            # Average TCAV score for the concept
            concept_importance[concept] = np.mean(tcav_scores)
        
        return concept_importance
    
    def run_stress_test(self, X_test: np.ndarray, stress_scenarios: Dict[str, np.ndarray]):
        """Run stress tests and analyze TCAV explanations."""
        results = {}
        
        for scenario_name, stress_multipliers in stress_scenarios.items():
            print(f"\n--- Stress Test: {scenario_name} ---")
            
            # Extend stress_multipliers to match X_test's feature count
            if len(stress_multipliers) != X_test.shape[1]:
                extended_multipliers = np.ones(X_test.shape[1])
                min_len = min(len(stress_multipliers), X_test.shape[1])
                extended_multipliers[:min_len] = stress_multipliers[:min_len]
                stress_multipliers = extended_multipliers
            
            X_stressed = X_test * stress_multipliers
            risk_prob_normal = self.risk_classifier.predict_proba(X_test)[:, 1]
            risk_prob_stressed = self.risk_classifier.predict_proba(X_stressed)[:, 1]
            risk_increase = risk_prob_stressed - risk_prob_normal
            
            tcav_importance = self.compute_tcav_concept_importance(X_stressed)
            
            results[scenario_name] = {
                'avg_risk_increase': np.mean(risk_increase),
                'max_risk_increase': np.max(risk_increase),
                'tcav_importance': tcav_importance,
                'highly_affected_counterparties': np.where(risk_increase > 0.1)[0]
            }
            
            print(f"Average risk increase: {np.mean(risk_increase):.3f}")
            print(f"Max risk increase: {np.max(risk_increase):.3f}")
            print("TCAV Concept Importance:")
            for concept, score in tcav_importance.items():
                print(f"  {concept}: {score:.3f}")
        
        return results
    
    def generate_explanations(self, counterparty_idx: int, X_test: np.ndarray, original_test_data: np.ndarray = None):
        """Generate TCAV-based explanations for a specific counterparty."""
        print(f"\n--- Risk Explanation for Counterparty {counterparty_idx} ---")
        
        # Get risk probability
        risk_prob = self.risk_classifier.predict_proba([X_test[counterparty_idx]])[0][1]
        print(f"Risk Probability: {risk_prob:.3f}")
        
        # Compute TCAV concept importance for the specific counterparty
        X_single = X_test[counterparty_idx:counterparty_idx+1]
        tcav_importance = self.compute_tcav_concept_importance(X_single)
        
        print("\nConcept-based Explanations (TCAV):")
        for concept, score in tcav_importance.items():
            print(f"  {concept}: {score:.3f}")
        
        # Feature-level analysis (using simple feature importance from RandomForest)
        feature_importance = self.risk_classifier.feature_importances_
        top_feature_indices = np.argsort(feature_importance)[-5:][::-1]
        
        print("\nTop Contributing Features (Random Forest Importance):")
        for i, idx in enumerate(top_feature_indices):
            if idx < len(self.feature_names):
                feature_name = self.feature_names[idx]
                importance_score = feature_importance[idx]
                
                if original_test_data is not None and idx < original_test_data.shape[1]:
                    feature_value = original_test_data[counterparty_idx, idx]
                    print(f"  {i+1}. {feature_name}: {importance_score:.3f} (Value: {feature_value:.2f})")
                else:
                    print(f"  {i+1}. {feature_name}: {importance_score:.3f}")
            else:
                print(f"  {i+1}. Quantum-derived feature {idx}: {feature_importance[idx]:.3f}")
        
        return {
            'risk_probability': risk_prob,
            'feature_importance': dict(zip(range(len(feature_importance)), feature_importance)),
            'concept_importance': tcav_importance
        }

def load_home_credit_data(data_path: str, n_samples: int = 5000) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load and preprocess Home Credit Default Risk dataset."""
    print("Loading Home Credit dataset...")
    
    # Verify file exists and is accessible
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Please check the file path.")
    if not os.access(data_path, os.R_OK):
        raise PermissionError(f"Permission denied for {data_path}. Check file permissions.")
    
    df = pd.read_csv(data_path)
    
    # Subsample to prevent slowdown
    if n_samples and len(df) > n_samples:
        df = df.sample(n=n_samples, random_state=42)
    
    # Select relevant features
    features = [
        'AMT_INCOME_TOTAL',  # Liquidity proxy
        'AMT_CREDIT',        # Credit exposure
        'AMT_ANNUITY',       # Repayment capacity
        'EXT_SOURCE_1',      # Credit score
        'EXT_SOURCE_2',      # Credit score
        'EXT_SOURCE_3',      # Credit score
    ]
    
    # Handle missing values
    for feature in features:
        if feature in df.columns:
            df[feature] = df[feature].fillna(df[feature].median())
        else:
            print(f"Warning: {feature} not found in dataset")
    
    # Filter features that exist in the dataset
    available_features = [f for f in features if f in df.columns]
    
    if not available_features:
        raise ValueError("None of the required features found in the dataset")
    
    X = df[available_features].values
    y = df['TARGET'].values
    
    print(f"Data shape: {X.shape}")
    print(f"Features used: {available_features}")
    print(f"High-risk counterparties: {np.sum(y)} ({np.mean(y)*100:.1f}%)")
    
    return X, y, available_features

def create_visualizations(model, X_test, y_test, feature_names, stress_results):
    """Create comprehensive visualizations."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Quantum-Inspired Causal XAI Results (TCAV)', fontsize=16)
    
    # 1. Feature Importance (RandomForest-based)
    feature_importance = model.risk_classifier.feature_importances_
    n_original = len(feature_names)
    feature_importance = feature_importance[:n_original]
    
    bars = axes[0, 0].bar(range(len(feature_names)), feature_importance, color='skyblue')
    axes[0, 0].set_xticks(range(len(feature_names)))
    axes[0, 0].set_xticklabels([name[:10] + "..." if len(name) > 10 else name for name in feature_names], 
                               rotation=45, ha='right')
    axes[0, 0].set_title('Random Forest Feature Importance (Original Features)')
    axes[0, 0].set_ylabel('Feature Importance')
    
    # Color bars by importance
    max_importance = max(feature_importance)
    for bar, importance in zip(bars, feature_importance):
        bar.set_color(plt.cm.Reds(importance / max_importance))
    
    # 2. Risk Distribution
    risk_probs = model.risk_classifier.predict_proba(X_test)[:, 1]
    y_pred = model.risk_classifier.predict(X_test)
    
    axes[0, 1].hist(risk_probs[y_test==0], bins=20, alpha=0.7, label='Actual Low Risk', color='green', density=True)
    axes[0, 1].hist(risk_probs[y_test==1], bins=20, alpha=0.7, label='Actual High Risk', color='red', density=True)
    axes[0, 1].axvline(x=0.5, color='black', linestyle='--', label='Decision Threshold')
    axes[0, 1].set_title('Risk Probability Distribution')
    axes[0, 1].set_xlabel('Predicted Risk Probability')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].legend()
    
    # 3. Stress Test Impact
    scenarios = list(stress_results.keys())
    avg_increases = [stress_results[s]['avg_risk_increase'] for s in scenarios]
    max_increases = [stress_results[s]['max_risk_increase'] for s in scenarios]
    
    x_pos = np.arange(len(scenarios))
    width = 0.35
    
    bars1 = axes[1, 0].bar(x_pos - width/2, avg_increases, width, label='Average Increase', color='orange', alpha=0.7)
    bars2 = axes[1, 0].bar(x_pos + width/2, max_increases, width, label='Maximum Increase', color='red', alpha=0.7)
    
    axes[1, 0].set_title('Stress Test Risk Impact')
    axes[1, 0].set_xlabel('Stress Test Scenarios')
    axes[1, 0].set_ylabel('Risk Increase')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels([s.replace('_', ' ').title() for s in scenarios], rotation=45, ha='right')
    axes[1, 0].legend()
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # 4. Concept Importance Heatmap (TCAV)
    concept_data = []
    concepts = ['liquidity', 'credit_exposure', 'credit_score']
    
    for scenario in scenarios:
        scenario_scores = stress_results[scenario]['tcav_importance']
        concept_data.append([scenario_scores.get(concept, 0) for concept in concepts])
    
    concept_df = pd.DataFrame(concept_data, 
                             index=[s.replace('_', ' ').title() for s in scenarios], 
                             columns=[c.replace('_', ' ').title() for c in concepts])
    
    sns.heatmap(concept_df, annot=True, cmap='RdYlBu_r', ax=axes[1, 1], fmt='.3f', cbar_kws={'label': 'TCAV Importance'})
    axes[1, 1].set_title('Concept Importance by Stress Scenario (TCAV)')
    axes[1, 1].set_xlabel('Financial Concepts')
    axes[1, 1].set_ylabel('Stress Test Scenarios')
    
    plt.tight_layout()
    plt.savefig('quantum_xai_results_tcav.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_summary(X, X_quantum, y_test, risk_probs, stress_results, model):
    """Print comprehensive summary statistics."""
    print("\n" + "="*60)
    print("QUANTUM-INSPIRED CAUSAL XAI SUMMARY (TCAV)")
    print("="*60)
    
    print(f"📊 Dataset Statistics:")
    print(f"   • Total samples processed: {len(X):,}")
    print(f"   • Original features: {X.shape[1]}")
    print(f"   • Quantum-embedded features: {X_quantum.shape[1]}")
    print(f"   • Feature expansion ratio: {X_quantum.shape[1]/X.shape[1]:.1f}x")
    
    print(f"\n🎯 Model Performance:")
    auc_score = roc_auc_score(y_test, risk_probs)
    print(f"   • AUC-ROC Score: {auc_score:.3f}")
    print(f"   • High-risk counterparties (>0.5): {np.sum(risk_probs > 0.5)}")
    print(f"   • Average risk probability: {np.mean(risk_probs):.3f}")
    
    print(f"\n🔗 Causal Discovery:")
    print(f"   • Causal relationships found: {model.causal_graph.number_of_edges()}")
    print(f"   • Features with causal connections: {model.causal_graph.number_of_nodes()}")
    
    print(f"\n⚡ Stress Test Results:")
    for scenario, results in stress_results.items():
        print(f"   • {scenario.replace('_', ' ').title()}:")
        print(f"     - Avg risk increase: {results['avg_risk_increase']:.3f}")
        print(f"     - Max risk increase: {results['max_risk_increase']:.3f}")
        print(f"     - Highly affected counterparties: {len(results['highly_affected_counterparties'])}")
        print(f"     - TCAV Concept Importance:")
        for concept, score in results['tcav_importance'].items():
            print(f"       * {concept}: {score:.3f}")
    
    print(f"\n✅ Pipeline completed successfully!")

def main_research_pipeline():
    """Main research pipeline with Home Credit dataset and TCAV explainability."""
    print("🧬 Quantum-Inspired Causal XAI for Counterparty Credit Risk")
    print("=" * 70)
    
    # Step 1: Load Home Credit dataset
    data_path = r'C:/Users/ASUS Vivobook/Desktop/Research Papers/Explainable AI/train.csv'
    try:
        X, y, feature_names = load_home_credit_data(data_path, n_samples=5000)
    except (FileNotFoundError, PermissionError) as e:
        print(f"Error: {e}")
        print("Please ensure the dataset file is accessible and correctly located.")
        return
    
    # Step 2: Initialize model
    print("\n2. Initializing Quantum-Inspired Causal XAI Model...")
    model = CounterpartyRiskModel(quantum_components=15, n_original_features=len(feature_names))
    
    # Step 3: Quantum-inspired feature embedding
    print("\n3. Applying quantum-inspired feature embedding...")
    X_quantum = model.prepare_features(X, feature_names)
    print(f"   Original features: {X.shape[1]}")
    print(f"   Quantum-embedded features: {X_quantum.shape[1]}")
    
    # Step 4: Causal discovery
    print("\n4. Discovering causal relationships...")
    causal_graph = model.discover_causal_relationships(X_quantum)
    print(f"   Causal graph: {causal_graph.number_of_nodes()} nodes, {causal_graph.number_of_edges()} edges")
    
    # Visualize causal graph
    model.causal_discovery.visualize_causal_graph(save_path='causal_graph.png')
    
    # Step 5: Train risk model
    print("\n5. Training counterparty risk model...")
    X_train, X_test, y_train, y_test = model.train_risk_model(X_quantum, y)
    
    # Split original data for explanations
    _, X_original_test, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Step 6: Setup TCAV concepts
    print("\n6. Setting up TCAV concepts for explainability...")
    model.setup_concepts_for_tcav(X_quantum, X)
    
    # Step 7: Stress testing
    print("\n7. Running stress tests...")
    # Create stress scenarios that match the number of features
    n_features = X.shape[1]
    stress_scenarios = {
        'interest_rate_shock': np.array([1.0, 2.0, 1.5, 1.2, 1.5, 0.8][:n_features] + [1.0] * max(0, n_features - 6)),
        'liquidity_crisis': np.array([0.3, 1.0, 1.5, 1.0, 1.0, 0.7][:n_features] + [1.0] * max(0, n_features - 6)),
        'credit_crunch': np.array([0.7, 1.0, 1.0, 2.0, 0.5, 0.5][:n_features] + [1.0] * max(0, n_features - 6)),
    }
    
    stress_results = model.run_stress_test(X_test, stress_scenarios)
    
    # Step 8: Generate explanations
    print("\n8. Generating explanations for specific counterparties...")
    risk_probs = model.risk_classifier.predict_proba(X_test)[:, 1]
    high_risk_indices = np.where(risk_probs > 0.5)[0]
    
    if len(high_risk_indices) > 0:
        explanation = model.generate_explanations(high_risk_indices[0], X_test, X_original_test)
    else:
        # Use highest risk counterparty
        highest_risk_idx = np.argmax(risk_probs)
        print(f"No counterparties with risk > 0.5. Using highest risk: {risk_probs[highest_risk_idx]:.3f}")
        explanation = model.generate_explanations(highest_risk_idx, X_test, X_original_test)
    
    # Step 9: Visualizations
    print("\n9. Generating visualizations...")
    create_visualizations(model, X_test, y_test, feature_names, stress_results)
    
    # Summary
    print_summary(X, X_quantum, y_test, risk_probs, stress_results, model)

if __name__ == "__main__":
    main_research_pipeline()