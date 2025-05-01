# =============================================
# IMPORTS
# =============================================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.datasets import make_classification
import warnings
from PIL import Image
import time
from cvxopt import matrix, solvers
import seaborn as sns

warnings.filterwarnings("ignore")
solvers.options['show_progress'] = False

# =============================================
# SVM IMPLEMENTATIONS
# =============================================

class HardMarginSVM:
    def __init__(self):
        self.w = None  # Weight vector
        self.b = None  # Bias term
        self.support_vectors = np.array([])
        self.support_labels = np.array([])
        self.alphas = np.array([])
        
    def fit(self, X, y):
        """
        Fit the hard margin SVM model to the training data.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training vectors
        y : array-like of shape (n_samples,)
            Target values (must be -1 or 1)
        """
        n_samples, n_features = X.shape
        
        try:
            # Quadratic programming problem setup
            P = matrix(np.outer(y, y) * np.dot(X, X.T) + 1e-8 * np.eye(n_samples))  # Add small diagonal for numerical stability
            q = matrix(-np.ones(n_samples))
            G = matrix(-np.eye(n_samples))
            h = matrix(np.zeros(n_samples))
            A = matrix(y.reshape(1, -1).astype('float'))
            b = matrix(0.0)
            
            # Solve QP problem
            solution = solvers.qp(P, q, G, h, A, b)
            alphas = np.array(solution['x']).flatten()
            
            # Get support vectors (alphas > 0)
            sv = alphas > 1e-5
            self.support_vectors = X[sv]
            self.support_labels = y[sv]
            self.alphas = alphas[sv]
            
            # Calculate weights
            self.w = np.sum((self.alphas * self.support_labels).reshape(-1, 1) * self.support_vectors, axis=0)
            
            # Calculate bias
            if len(self.support_vectors) > 0:
                self.b = np.mean(self.support_labels - np.dot(self.support_vectors, self.w))
            else:
                self.b = 0
                
        except Exception as e:
            st.error(f"Error in HardMarginSVM: {str(e)}")
            # Fallback to least squares solution if QP fails
            self.w = np.linalg.lstsq(X, y, rcond=None)[0]
            self.b = 0
    
    def decision_function(self, X):
        """
        Calculate the decision function value for each sample.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input samples
            
        Returns:
        --------
        array of shape (n_samples,)
            Decision function values
        """
        if self.w is None or self.b is None:
            raise ValueError("Model not trained yet. Call fit() first.")
        return np.dot(X, self.w) + self.b
    
    def predict(self, X):
        """
        Predict class labels for samples in X.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input samples
            
        Returns:
        --------
        array of shape (n_samples,)
            Predicted class labels (-1 or 1)
        """
        return np.sign(self.decision_function(X))

class SoftMarginSVM:
    def __init__(self, C=1.0, optimization='qp'):
        self.C = C
        self.optimization = optimization
        self.w = None
        self.b = None
        self.support_vectors = np.array([])
        self.support_labels = np.array([])
        self.alphas = np.array([])
        self.loss_history = []
        
    def fit(self, X, y, lr=0.01, epochs=1000):
        n_samples, n_features = X.shape
        
        try:
            if self.optimization == 'qp':
                # Add regularization to P matrix
                P = matrix(np.outer(y, y) * np.dot(X, X.T) + 1e-8 * np.eye(n_samples))
                q = matrix(-np.ones(n_samples))
                G = matrix(np.vstack((-np.eye(n_samples), np.eye(n_samples))))
                h = matrix(np.hstack((np.zeros(n_samples), np.ones(n_samples) * self.C)))
                A = matrix(y.reshape(1, -1).astype('float'))
                b = matrix(0.0)
                
                solution = solvers.qp(P, q, G, h, A, b)
                alphas = np.array(solution['x']).flatten()
                
                sv = (alphas > 1e-5) & (alphas < self.C - 1e-5)
                self.support_vectors = X[sv]
                self.support_labels = y[sv]
                self.alphas = alphas[sv]
                
                self.w = np.sum((alphas * y).reshape(-1, 1) * X, axis=0)
                
                if len(self.support_vectors) > 0:
                    self.b = np.mean(y[sv] - np.dot(X[sv], self.w))
                else:
                    self.b = 0
                    
                    
            elif self.optimization in ['gd', 'subgrad']:
                # Initialize weights
                self.w = np.zeros(n_features)
                self.b = 0
                n = len(y)
                
                for epoch in range(epochs):
                    margins = y * (np.dot(X, self.w) + self.b)
                    grad_w = np.zeros(n_features)
                    grad_b = 0
                    
                    # Calculate gradients
                    for i in range(n):
                        if margins[i] < 1:
                            grad_w += -y[i] * X[i]
                            grad_b += -y[i]
                    
                    grad_w = self.w + (self.C * grad_w / n)
                    grad_b = (self.C * grad_b / n)
                    
                    # Update weights
                    if self.optimization == 'gd':
                        self.w -= lr * grad_w
                        self.b -= lr * grad_b
                    else:  # subgrad
                        self.w -= (lr/np.sqrt(epoch+1)) * grad_w
                        self.b -= (lr/np.sqrt(epoch+1)) * grad_b
                    
                    # Calculate and store loss
                    loss = 0.5 * np.dot(self.w, self.w) + self.C * np.sum(np.maximum(0, 1 - margins))
                    self.loss_history.append(loss)
                
                # For GD methods, all points are potential support vectors
                self.support_vectors = X
                self.support_labels = y
                
        except Exception as e:
            st.error(f"Error in SoftMarginSVM: {str(e)}")
            self.w = np.zeros(X.shape[1])
            self.b = 0
        
    def predict(self, X):
        if self.w is None:
            return np.zeros(len(X))
        return np.sign(np.dot(X, self.w) + self.b)
     
    def plot_confusion_matrix(self, X, y):
        """Plot confusion matrix for test data"""
        y_pred = self.predict(X)
        cm = confusion_matrix(y, y_pred)
        
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['-1', '1'], yticklabels=['-1', '1'])
        ax.set_title('Confusion Matrix')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        st.pyplot(fig)
class KernelSVM:
    def __init__(self, kernel='rbf', C=1.0, gamma='scale', degree=3, coef0=0.0, 
                 learning_rate=0.01, max_iter=1000, tol=1e-3, optimizer='qp', 
                 random_state=None, verbose=False):
        """
        Robust Kernel SVM implementation with multiple optimization methods.
        
        Parameters:
        -----------
        kernel : {'linear', 'rbf', 'poly', 'sigmoid'}, default='rbf'
            Kernel function type
        C : float, default=1.0
            Regularization parameter
        gamma : {'scale', 'auto'} or float, default='scale'
            Kernel coefficient for 'rbf', 'poly' and 'sigmoid'
        degree : int, default=3
            Degree for polynomial kernel
        coef0 : float, default=0.0
            Independent term in polynomial and sigmoid kernels
        learning_rate : float, default=0.01
            Learning rate for gradient-based optimizers
        max_iter : int, default=1000
            Maximum number of iterations
        tol : float, default=1e-3
            Tolerance for stopping criterion
        optimizer : {'qp', 'sgd', 'subgrad'}, default='qp'
            Optimization method
        random_state : int or None, default=None
            Random seed for reproducibility
        verbose : bool, default=False
            Whether to print optimization progress
        """
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.optimizer = optimizer
        self.random_state = random_state
        self.verbose = verbose
        self.alpha = np.array([])
        self.b = 0
        self.X_train = np.array([])
        self.y_train = np.array([])
        self.support_vectors = np.array([])
        self.support_labels = np.array([])
        self.loss_history = []
        self._gamma_value = None
        self._alpha_history = []
        self._b_history = []
        
    def _initialize_gamma(self, X):
        """Initialize gamma value based on input data with validation"""
        if isinstance(self.gamma, str):
            if self.gamma == 'scale':
                if X.shape[1] == 0:
                    self._gamma_value = 1.0
                else:
                    var = X.var()
                    self._gamma_value = 1.0 / (X.shape[1] * var) if var != 0 else 1.0
            elif self.gamma == 'auto':
                self._gamma_value = 1.0 / X.shape[1] if X.shape[1] > 0 else 1.0
            else:
                raise ValueError(f"Invalid gamma value: {self.gamma}")
        else:
            self._gamma_value = float(self.gamma)
            if self._gamma_value <= 0:
                raise ValueError("Gamma must be positive")
    
    def _validate_data(self, X, y):
        """Validate input data and convert to numpy arrays"""
        X = np.asarray(X)
        y = np.asarray(y)
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
        if len(X.shape) != 2:
            raise ValueError("X must be a 2D array")
        if len(np.unique(y)) < 2:
            raise ValueError("Need samples from at least two classes")
        return X, np.where(y <= 0, -1, 1)
    
    def _linear_kernel(self, X1, X2):
        """Linear kernel with numerical stability checks"""
        return np.dot(X1, X2.T)
    
    def _rbf_kernel(self, X1, X2):
        """RBF kernel with numerical stability"""
        X1_sq = np.sum(X1**2, axis=1)
        X2_sq = np.sum(X2**2, axis=1)
        dist = X1_sq[:, np.newaxis] + X2_sq - 2 * np.dot(X1, X2.T)
        return np.exp(-self._gamma_value * np.clip(dist, 0, None))
    
    def _poly_kernel(self, X1, X2):
        """Polynomial kernel with validation"""
        if self.degree < 1:
            raise ValueError("Degree must be >= 1")
        return (self._gamma_value * np.dot(X1, X2.T) + self.coef0) ** self.degree
    
    def _sigmoid_kernel(self, X1, X2):
        """Sigmoid kernel with numerical stability"""
        K = np.dot(X1, X2.T)
        np.clip(K, -500, 500, out=K)  # Prevent overflow
        return np.tanh(self._gamma_value * K + self.coef0)
    
    def _kernel(self, X1, X2):
        """Dispatch to appropriate kernel function with error handling"""
        try:
            if self.kernel == 'linear':
                return self._linear_kernel(X1, X2)
            elif self.kernel == 'rbf':
                return self._rbf_kernel(X1, X2)
            elif self.kernel == 'poly':
                return self._poly_kernel(X1, X2)
            elif self.kernel == 'sigmoid':
                return self._sigmoid_kernel(X1, X2)
            else:
                raise ValueError(f"Unknown kernel: {self.kernel}")
        except Exception as e:
            raise RuntimeError(f"Kernel computation failed: {str(e)}")
    
    def fit(self, X, y):
        """Fit the SVM model with robust error handling"""
        try:
            X, y = self._validate_data(X, y)
            self._initialize_gamma(X)
            self.X_train = X
            self.y_train = y
            
            if self.random_state is not None:
                np.random.seed(self.random_state)
            
            if self.optimizer == 'qp':
                self._fit_qp(X, y)
            elif self.optimizer in ['sgd', 'subgrad']:
                self._fit_gradient(X, y)
            else:
                raise ValueError(f"Unknown optimizer: {self.optimizer}")
                
        except Exception as e:
            raise RuntimeError(f"Failed to fit model: {str(e)}")
        
        return self
    
    def _fit_qp(self, X, y):
        """Quadratic programming solver with enhanced stability"""
        n_samples = X.shape[0]
        
        try:
            # Regularized kernel matrix
            K = self._kernel(X, X) + 1e-8 * np.eye(n_samples)
            
            # QP matrices
            P = matrix(np.outer(y, y) * K)
            q = matrix(-np.ones(n_samples))
            G = matrix(np.vstack((-np.eye(n_samples), np.eye(n_samples))))
            h = matrix(np.hstack((np.zeros(n_samples), np.ones(n_samples) * self.C)))
            A = matrix(y.reshape(1, -1).astype('float'))
            b = matrix(0.0)
            
            # Solver configuration
            solvers.options = {
                'show_progress': self.verbose,
                'maxiters': 500,
                'abstol': 1e-7,
                'reltol': 1e-6,
                'feastol': 1e-7
            }
            
            solution = solvers.qp(P, q, G, h, A, b)
            alphas = np.array(solution['x']).flatten()
            
            # Get support vectors
            sv = (alphas > 1e-5)
            if not np.any(sv):
                raise ValueError("QP solution produced no support vectors")
                
            self.support_vectors = X[sv]
            self.support_labels = y[sv]
            self.alpha = alphas[sv]
            
            # Calculate bias
            K_sv = self._kernel(self.support_vectors, self.support_vectors)
            self.b = np.mean(self.support_labels - np.sum(self.alpha * self.support_labels * K_sv, axis=1))
            
        except Exception as e:
            if self.verbose:
                print(f"QP solver failed (reason: {str(e)}), switching to gradient descent")
            self.optimizer = 'subgrad'
            self._fit_gradient(X, y)
    
    def _fit_gradient(self, X, y):
        """Gradient-based optimization with improved convergence"""
        n_samples = X.shape[0]
        self.alpha = np.zeros(n_samples)
        self.b = 0
        K = self._kernel(X, X)
        
        best_loss = float('inf')
        best_alpha = self.alpha.copy()
        best_b = self.b
        
        for epoch in range(self.max_iter):
            indices = np.random.permutation(n_samples)
            
            for i in indices:
                f_i = np.sum(self.alpha * y * K[:, i]) + self.b
                
                if y[i] * f_i < 1:
                    lr = (self.learning_rate / np.sqrt(epoch + 1) 
                          if self.optimizer == 'subgrad' 
                          else self.learning_rate)
                    
                    grad_alpha = 1 - y[i] * y * K[i, :]
                    self.alpha -= lr * grad_alpha
                    self.alpha = np.clip(self.alpha, 0, self.C)
                    self.b -= lr * (-y[i])
            
            # Store parameters for accuracy curve
            self._alpha_history.append(self.alpha.copy())
            self._b_history.append(self.b)
            
            current_loss = self._compute_loss(K, y)
            self.loss_history.append(current_loss)
            
            if current_loss < best_loss:
                best_loss = current_loss
                best_alpha = self.alpha.copy()
                best_b = self.b
                no_improvement = 0
            else:
                no_improvement += 1
                
            if self.verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {current_loss:.4f}")
                
            if no_improvement >= 10 or (epoch > 10 and abs(self.loss_history[-2] - current_loss) < self.tol):
                if self.verbose:
                    print(f"Early stopping at epoch {epoch}")
                break
        
        # Store best parameters
        self.alpha = best_alpha
        self.b = best_b
        
        # Extract support vectors
        sv = (self.alpha > 1e-5)
        self.support_vectors = X[sv]
        self.support_labels = y[sv]
        self.alpha = self.alpha[sv]
    
    def _compute_loss(self, K, y):
        """Compute regularized SVM loss"""
        margins = y * (np.dot(K, self.alpha * y) + self.b)
        hinge_loss = np.sum(np.maximum(0, 1 - margins))
        reg_loss = 0.5 * np.dot(self.alpha, np.dot(K, self.alpha))
        return reg_loss + self.C * hinge_loss
    
    def decision_function(self, X):
        """Calculate signed distance to hyperplane"""
        X = np.asarray(X)
        if len(self.support_vectors) == 0:
            return np.zeros(len(X))
        kernel_vals = self._kernel(X, self.support_vectors)
        return np.sum(self.alpha * self.support_labels * kernel_vals, axis=1) + self.b
    
    def predict(self, X):
        """Predict class labels (-1 or 1)"""
        return np.sign(self.decision_function(X))
    
    def predict_proba(self, X):
        """Predict probability estimates (not properly calibrated)"""
        dec_values = self.decision_function(X)
        proba = 1.0 / (1.0 + np.exp(-dec_values))
        return np.vstack((1-proba, proba)).T
    
    def score(self, X, y):
        """Return mean accuracy on test data"""
        y_pred = self.predict(X)
        y_true = np.where(np.asarray(y) <= 0, -1, 1)
        return np.mean(y_pred == y_true)
        
    def plot_loss_curve(self):
        """Plot the training loss curve"""
        if not self.loss_history:
            st.warning("No loss history available. Run fit() with gradient-based optimizer first.")
            return
            
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(self.loss_history, label='Training Loss')
        ax.set_title('Training Loss Curve')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)
    
    def plot_accuracy_curve(self, X, y, eval_every=10):
        """Plot accuracy during training (only for gradient methods)"""
        if self.optimizer == 'qp':
            st.warning("Accuracy curve only available for gradient-based optimizers")
            return
            
        if not self.loss_history:
            st.warning("No training history available")
            return
            
        accuracies = []
        for i in range(0, len(self.loss_history), eval_every):
            temp_alpha = self._alpha_history[i] if hasattr(self, '_alpha_history') else self.alpha
            temp_b = self._b_history[i] if hasattr(self, '_b_history') else self.b
            y_pred = self._predict_with_params(X, temp_alpha, temp_b)
            acc = accuracy_score(y, y_pred)
            accuracies.append(acc)
            
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(range(0, len(self.loss_history), eval_every), accuracies)
        ax.set_title('Training Accuracy Curve')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Accuracy')
        ax.grid(True)
        st.pyplot(fig)
    
    def plot_confusion_matrix(self, X, y):
        """Plot confusion matrix for test data"""
        y_pred = self.predict(X)
        cm = confusion_matrix(y, y_pred)
        
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['-1', '1'], yticklabels=['-1', '1'])
        ax.set_title('Confusion Matrix')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        st.pyplot(fig)
        
    def _predict_with_params(self, X, alpha, b):
        """Helper function for accuracy curve calculation"""
        if len(self.support_vectors) == 0:
            return np.zeros(len(X))
        kernel_vals = self._kernel(X, self.support_vectors)
        return np.sign(np.sum(alpha * self.support_labels * kernel_vals, axis=1) + b)

class SVMAnalyzer:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.results = {}
    
    def compare_methods(self, methods=None, kernel='rbf', C=1.0):
        """Compare different optimization methods"""
        if methods is None:
            methods = ['qp', 'subgrad']  # Default methods if none provided
            
        for method in methods:
            svm = KernelSVM(kernel=kernel, C=C, optimizer=method, verbose=False)
            svm.fit(self.X_train, self.y_train)
            
            train_acc = svm.score(self.X_train, self.y_train)
            test_acc = svm.score(self.X_test, self.y_test)
            convergence = len(svm.loss_history) if hasattr(svm, 'loss_history') and svm.loss_history else 0
            
            self.results[method] = {
                'model': svm,
                'train_accuracy': train_acc,
                'test_accuracy': test_acc,
                'convergence_iterations': convergence,
                'final_loss': svm.loss_history[-1] if hasattr(svm, 'loss_history') and svm.loss_history else None,
                'loss_history': svm.loss_history if hasattr(svm, 'loss_history') else None
            }
    
    def plot_comparison(self):
        """Visual comparison of methods"""
        if not self.results:
            st.warning("Run compare_methods() first")
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Accuracy comparison
        methods = list(self.results.keys())
        train_acc = [self.results[m]['train_accuracy'] for m in methods]
        test_acc = [self.results[m]['test_accuracy'] for m in methods]
        
        x = range(len(methods))
        ax1.bar(x, train_acc, width=0.4, label='Train Accuracy')
        ax1.bar([i + 0.4 for i in x], test_acc, width=0.4, label='Test Accuracy')
        ax1.set_xticks([i + 0.2 for i in x])
        ax1.set_xticklabels(methods)
        ax1.set_title('Accuracy Comparison')
        ax1.set_ylim(0, 1.1)
        ax1.legend()
        
        # Loss curves comparison
        for method in methods:
            if self.results[method]['loss_history']:
                ax2.plot(self.results[method]['loss_history'], label=f'{method} loss')
        ax2.set_title('Loss Convergence')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Loss')
        ax2.legend()
        
        st.pyplot(fig)
    
    def generate_report(self):
        """Generate textual analysis report"""
        st.subheader("SVM Optimization Methods Comparison")
        for method, data in self.results.items():
            with st.expander(f"Method: {method.upper()}"):
                st.write(f"- Training Accuracy: {data['train_accuracy']:.4f}")
                st.write(f"- Test Accuracy: {data['test_accuracy']:.4f}")
                st.write(f"- Generalization Gap: {data['train_accuracy']-data['test_accuracy']:.4f}")
                
                if data['convergence_iterations']:
                    st.write(f"- Converged in {data['convergence_iterations']} iterations")
                    st.write(f"- Final Loss: {data['final_loss']:.4f}")
                
                # Stability analysis
                if data['loss_history'] and len(data['loss_history']) > 10:
                    last_10_loss = data['loss_history'][-10:]
                    loss_var = np.var(last_10_loss)
                    st.write(f"- Loss Stability (last 10 iters variance): {loss_var:.6f}")

# =============================================
# HELPER FUNCTIONS
# =============================================

def load_data(uploaded_file):
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    else:
        # Default dataset if none uploaded
        X, y = make_classification(
            n_samples=100, 
            n_features=2, 
            n_classes=2, 
            n_clusters_per_class=1, 
            random_state=42,
            n_informative=2,  # Explicitly set to match n_features
            n_redundant=0,    # Set to 0 since we only have 2 features
            n_repeated=0      # Set to 0 since we only have 2 features
        )
        df = pd.DataFrame(X, columns=['feature_1', 'feature_2'])
        df['target'] = y
    return df
    
def plot_feature_pairs(X, y, features):
    feature_pairs = list(itertools.combinations(range(len(features)), 2))
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    for i, (f1, f2) in enumerate(feature_pairs[:6]):
        ax = axes[i]
        ax.scatter(X[:, f1], X[:, f2], c=y, cmap='bwr', alpha=0.6)
        ax.set_xlabel(features[f1])
        ax.set_ylabel(features[f2])
        ax.set_title(f"{features[f1]} vs {features[f2]}")
    plt.tight_layout()
    return fig

def plot_decision_boundary(model, X, y, features, pair=(0,1)):
    x_min, x_max = X[:, pair[0]].min() - 1, X[:, pair[0]].max() + 1
    y_min, y_max = X[:, pair[1]].min() - 1, X[:, pair[1]].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    
    grid_points = np.zeros((xx.ravel().shape[0], X.shape[1]))
    grid_points[:, pair[0]] = xx.ravel()
    grid_points[:, pair[1]] = yy.ravel()
    
    Z = model.predict(grid_points)
    Z = Z.reshape(xx.shape)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.contourf(xx, yy, Z, alpha=0.2, cmap='bwr')
    ax.scatter(X[:, pair[0]], X[:, pair[1]], c=y, cmap='bwr', alpha=0.6)
    ax.set_xlabel(features[pair[0]])
    ax.set_ylabel(features[pair[1]])
    ax.set_title("Decision Boundary")
    return fig

def calculate_hinge_loss(X, y, w, b, C):
    margins = y * (np.dot(X, w) + b)
    hinge_loss = np.mean(np.maximum(0, 1 - margins))
    reg_loss = 0.5 * np.dot(w, w)
    return hinge_loss + C * reg_loss

# =============================================
# MAIN APP
# =============================================

def main():
    # Add title and image
    st.set_page_config(page_title="SVM Analyzer🤖💻🧠", layout="wide", page_icon="🤖")
    st.title("🤖 SVM Model Analyzer (From Scratch)")
    st.image(r"C:\Users\ranee\Dropbox\PC\Desktop\convex\SVM1.jpeg", width=1000)
    
    with st.sidebar:
        # Title with emoji and divider
        st.title("⚙️ Settings")
        st.markdown("---")  # Horizontal line
        
        # Team name with emoji and emphasis
        st.subheader("👨‍💻 Team Name")
        st.markdown("### 🚀 *The 8 Optimizers*")  # Italicized for flair
        st.image(r"C:\Users\ranee\Dropbox\PC\Desktop\convex\team logo.png")
        
        # Team information
        st.header("👨‍💻 Team Members")
        team_members = [
            {"name": "Raneen Ashraf Yehia", "id": "22010091"},
            {"name": "Rewan Gaber Khalaf", "id": "22010093"},
            {"name": "Mariam Mohamed Ramdan", "id": "22010253"},
            {"name": "Abdelruhman Salah Anwar", "id": "20221458503"},
            {"name": "Omar Mohamed Mostafa", "id": "2022446471"},
            {"name": "Mahmoud Reda Hassan Nour", "id": "20221469438"},
            {"name": "Abubakr Mohamed Mahmoud ", "id": "20221458962"},
            {"name": "Fares Mohamed Fathy", "id": "20221461330"}
        ]
        
        for member in team_members:
            with st.expander(f"👤 {member['name']}"):
                st.write(f"**ID:** {member['id']}")
        
        # Divider
        st.markdown("---")
        
        # ----------------------------
        # 1. DATA UPLOAD SECTION
        # ----------------------------
        st.header("📤 1. Data Upload")
        uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx", "xls"], key="file_uploader")
        
        df = load_data(uploaded_file)
        
        if uploaded_file is not None:
            # Let user select target column
            target_col = st.selectbox("Select target column", df.columns, index=len(df.columns)-1)
            # Convert labels to {-1, 1}
            y = np.where(df[target_col].values == 0, -1, 1)
            X = df.drop(columns=[target_col]).values
        else:
            y = np.where(df['target'].values == 0, -1, 1)
            X = df.drop(columns=['target']).values
            target_col = 'target'

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        features = df.drop(columns=[target_col]).columns.tolist()

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # ----------------------------
    # 2. DATA VISUALIZATION SECTION
    # ----------------------------
    st.header("📊 2. Data Visualization")
    
    # Show basic info
    col1, col2 = st.columns(2)
    with col1:
        st.write("Data Preview:")
        st.dataframe(df.head())
    
    with col2:
        st.write("Class Distribution:")
        fig, ax = plt.subplots()
        df[target_col].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax)
        st.pyplot(fig)
    
    # Feature visualization
    st.subheader("Feature Relationships")
    
    if len(features) >= 2:
        selected_x = st.selectbox("X-axis feature", features, index=0, key="x_axis_feature")
        selected_y = st.selectbox("Y-axis feature", features, index=1 if len(features) > 1 else 0, key="y_axis_feature")
        
        fig, ax = plt.subplots()
        ax.scatter(df[selected_x], df[selected_y], c=df[target_col], cmap='bwr', alpha=0.6)
        ax.set_xlabel(selected_x)
        ax.set_ylabel(selected_y)
        ax.set_title(f"{selected_x} vs {selected_y}")
        st.pyplot(fig)
        
        # Plot all feature pairs
        st.write("Pairwise feature relationships:")
        fig = plot_feature_pairs(X_scaled, y, features)
        st.pyplot(fig)

    # ----------------------------
    # 3. MODEL TRAINING SECTION
    # ----------------------------
    st.header("🤖 3. SVM Implementations from Scratch")

    tab1, tab2, tab3 = st.tabs(["Linear SVMs", "Kernel SVMs", "Model Comparison"])

    # =============================================
    # LINEAR SVMs TAB
    # =============================================
    with tab1:
        st.subheader("Linear SVM Models")
        
        # Create columns for the two linear models
        col_linear1, col_linear2 = st.columns(2)
        
        with col_linear1:
            st.markdown("#### 🔒 Hard Margin SVM")
            st.caption("Maximizes margin with strict classification (no misclassification allowed)")
            if st.button("Train Hard Margin SVM", key='train_hard', help="Best for perfectly separable data"):
                with st.spinner("Training in progress..."):
                    start_time = time.time()
                    model = HardMarginSVM()
                    model.fit(X_train, y_train)
                    
                    train_acc = accuracy_score(y_train, model.predict(X_train))
                    val_acc = accuracy_score(y_val, model.predict(X_val))
                    training_time = time.time() - start_time
                    
                    st.success(f"""
                    Training Results:
                    - Training Accuracy: {train_acc:.4f}
                    - Validation Accuracy: {val_acc:.4f}
                    - Training Time: {training_time:.2f} seconds
                    - Number of Support Vectors: {len(model.support_vectors)}
                    """)
                    
                    # Plot decision boundary
                    if len(features) >= 2:
                        fig = plot_decision_boundary(model, X_train, y_train, features)
                        st.pyplot(fig)
                    
                    # Show support vectors
                    st.subheader("Support Vectors")
                    if len(model.support_vectors) > 0:
                        sv_df = pd.DataFrame(model.support_vectors, columns=features)
                        sv_df['Label'] = model.support_labels
                        st.dataframe(sv_df)
                    else:
                        st.warning("No support vectors found")
        
        with col_linear2:
            st.markdown("#### 🧈 Soft Margin SVM")
            st.caption("Allows some misclassification for better generalization")
            
            # Soft margin parameters
            C = st.slider("Regularization (C)", 0.01, 10.0, 1.0, key='C_slider')
            optimization = st.selectbox(
                "Optimization Method",
                ["Quadratic Programming", "Gradient Descent", "Subgradient Descent"],
                key='optimization_select'
            )
            
            if optimization != "Quadratic Programming":
                lr = st.slider("Learning Rate", 0.001, 0.1, 0.01, key='lr_slider')
                epochs = st.slider("Epochs", 100, 5000, 1000, step=100, key='epochs_slider')
                tolerance = st.slider("Convergence Tolerance", 0.0001, 0.01, 0.001, step=0.0001, key='tolerance_slider')
            
            if st.button("Train Soft Margin SVM", key='train_soft'):
                with st.spinner("Training in progress..."):
                    start_time = time.time()
                    
                    opt_map = {
                        "Quadratic Programming": "qp",
                        "Gradient Descent": "gd",
                        "Subgradient Descent": "subgrad"
                    }
                    
                    model = SoftMarginSVM(
                        C=C,
                        optimization=opt_map[optimization]
                    )
                    
                    if optimization == "Quadratic Programming":
                        model.fit(X_train, y_train)
                    else:
                        model.fit(X_train, y_train, lr=lr, epochs=epochs)
                    
                    # Calculate predictions and metrics
                    y_train_pred = model.predict(X_train)
                    y_val_pred = model.predict(X_val)
                    y_train_binary = np.where(y_train == -1, 0, 1)
                    y_val_binary = np.where(y_val == -1, 0, 1)
                    y_train_pred_binary = np.where(y_train_pred == -1, 0, 1)
                    y_val_pred_binary = np.where(y_val_pred == -1, 0, 1)
                    
                    train_acc = accuracy_score(y_train_binary, y_train_pred_binary)
                    val_acc = accuracy_score(y_val_binary, y_val_pred_binary)
                    training_time = time.time() - start_time
                    
                    train_hinge = calculate_hinge_loss(X_train, y_train, model.w, model.b, C)
                    val_hinge = calculate_hinge_loss(X_val, y_val, model.w, model.b, C)
                    
                    # Display results
                    st.success(f"""
                    Training Results:
                    - Training Accuracy: {train_acc:.4f}
                    - Validation Accuracy: {val_acc:.4f}
                    - Training Hinge Loss: {train_hinge:.4f}
                    - Validation Hinge Loss: {val_hinge:.4f}
                    - Training Time: {training_time:.2f} seconds
                    - Number of Support Vectors: {len(model.support_vectors) if hasattr(model, 'support_vectors') else 'N/A'}
                    """)
                    
                    # Plot decision boundary
                    if len(features) >= 2:
                        fig = plot_decision_boundary(model, X_train, y_train, features)
                        st.pyplot(fig)
                    
                    # Add visualizations for gradient methods
                    if optimization != "Quadratic Programming" and hasattr(model, 'loss_history'):
                        viz_tab1, viz_tab2 = st.tabs(["Loss Curve", "Confusion Matrix"])
                        
                        with viz_tab1:
                            fig, ax = plt.subplots(figsize=(8, 5))
                            ax.plot(model.loss_history)
                            ax.set_title("Training Loss Curve")
                            ax.set_xlabel("Iteration")
                            ax.set_ylabel("Loss")
                            st.pyplot(fig)
                        
                        with viz_tab2:
                            # Convert y_val to binary (0,1) for consistency
                            y_val_binary = np.where(y_val == -1, 0, 1)
                            y_pred_binary = np.where(model.predict(X_val) == -1, 0, 1)
                            model.plot_confusion_matrix(X_val, y_val_binary)

    # =============================================
    # KERNEL SVMs TAB
    # =============================================
    with tab2:
        st.subheader("Kernel SVM Models")
        
        # Create columns for the three kernel models
        col_kernel1, col_kernel2, col_kernel3 = st.columns(3)
        
        with col_kernel1:
            st.markdown("#### 🌐 RBF Kernel SVM")
            st.caption("Uses Gaussian radial basis functions for non-linear boundaries")
            
            C_rbf = st.slider("Regularization (C)", 0.01, 10.0, 1.0, key='rbf_C_slider')
            gamma_rbf = st.slider("Gamma", 0.001, 1.0, 0.1, key='rbf_gamma_slider')
            
            # Updated optimizer selection
            optimizer_rbf = st.selectbox("Optimization Method", 
                                       ["Quadratic Programming", "Gradient Descent", "Subgradient Descent"], 
                                       key='rbf_optimizer')
            
            # Show parameters for iterative methods
            if optimizer_rbf != "Quadratic Programming":
                lr_rbf = st.slider("Learning Rate", 0.001, 0.1, 0.01, key='rbf_lr')
                max_iter_rbf = st.slider("Max Iterations", 100, 5000, 1000, key='rbf_max_iter')
                tol_rbf = st.slider("Convergence Tolerance", 0.0001, 0.01, 0.001, key='rbf_tol')
            
            if st.button("Train RBF Kernel SVM", key='train_rbf'):
                with st.spinner("Training in progress..."):
                    start_time = time.time()
                    
                    # Map optimizer names to internal codes
                    opt_map = {
                        "Quadratic Programming": "qp",
                        "Gradient Descent": "sgd",
                        "Subgradient Descent": "subgrad"
                    }
                    
                    model = KernelSVM(
                        kernel='rbf', 
                        C=C_rbf, 
                        gamma=gamma_rbf,
                        optimizer=opt_map[optimizer_rbf],
                        learning_rate=lr_rbf if optimizer_rbf != "Quadratic Programming" else 0.01,
                        max_iter=max_iter_rbf if optimizer_rbf != "Quadratic Programming" else 1000,
                        tol=tol_rbf if optimizer_rbf != "Quadratic Programming" else 1e-3
                    )
                    model.fit(X_train, y_train)
                    
                    train_acc = accuracy_score(y_train, model.predict(X_train))
                    val_acc = accuracy_score(y_val, model.predict(X_val))
                    training_time = time.time() - start_time
                    
                    # Create results display
                    result_text = f"""
                    Training Results:
                    - Training Accuracy: {train_acc:.4f}
                    - Validation Accuracy: {val_acc:.4f}
                    - Training Time: {training_time:.2f} seconds
                    - Number of Support Vectors: {len(model.support_vectors)}
                    - Optimization Method: {optimizer_rbf}
                    """
                    
                    st.success(result_text)
                    
                    # Add visualization tabs
                    viz_tab1, viz_tab2, viz_tab3 = st.tabs(["Loss Curve", "Accuracy", "Confusion Matrix"])
                    
                    with viz_tab1:
                        model.plot_loss_curve()
                    
                    with viz_tab2:
                        model.plot_accuracy_curve(X_val, y_val)
                    
                    with viz_tab3:
                        model.plot_confusion_matrix(X_val, y_val)
                    
                    if len(features) >= 2:
                        fig = plot_decision_boundary(model, X_train, y_train, features)
                        st.pyplot(fig)
        
        with col_kernel2:
            st.markdown("#### 🔺 Polynomial Kernel SVM")
            st.caption("Captures polynomial relationships between features")
            
            C_poly = st.slider("Regularization (C)", 0.01, 10.0, 1.0, key='poly_C_slider')
            degree_poly = st.slider("Degree", 2, 5, 3, key='poly_degree_slider')
            gamma_poly = st.slider("Gamma", 0.001, 1.0, 0.1, key='poly_gamma_slider')
            
            # Updated optimizer selection
            optimizer_poly = st.selectbox("Optimization Method", 
                                        ["Quadratic Programming", "Gradient Descent", "Subgradient Descent"], 
                                        key='poly_optimizer')
            
            if optimizer_poly != "Quadratic Programming":
                lr_poly = st.slider("Learning Rate", 0.001, 0.1, 0.01, key='poly_lr')
                max_iter_poly = st.slider("Max Iterations", 100, 5000, 1000, key='poly_max_iter')
                tol_poly = st.slider("Convergence Tolerance", 0.0001, 0.01, 0.001, key='poly_tol')
            
            if st.button("Train Polynomial Kernel SVM", key='train_poly'):
                with st.spinner("Training in progress..."):
                    start_time = time.time()
                    
                    opt_map = {
                        "Quadratic Programming": "qp",
                        "Gradient Descent": "sgd",
                        "Subgradient Descent": "subgrad"
                    }
                    
                    model = KernelSVM(
                        kernel='poly', 
                        C=C_poly, 
                        degree=degree_poly, 
                        gamma=gamma_poly,
                        optimizer=opt_map[optimizer_poly],
                        learning_rate=lr_poly if optimizer_poly != "Quadratic Programming" else 0.01,
                        max_iter=max_iter_poly if optimizer_poly != "Quadratic Programming" else 1000,
                        tol=tol_poly if optimizer_poly != "Quadratic Programming" else 1e-3
                    )
                    model.fit(X_train, y_train)
                    
                    train_acc = accuracy_score(y_train, model.predict(X_train))
                    val_acc = accuracy_score(y_val, model.predict(X_val))
                    training_time = time.time() - start_time
                    
                    result_text = f"""
                    Training Results:
                    - Training Accuracy: {train_acc:.4f}
                    - Validation Accuracy: {val_acc:.4f}
                    - Training Time: {training_time:.2f} seconds
                    - Number of Support Vectors: {len(model.support_vectors)}
                    - Optimization Method: {optimizer_poly}
                    """
                    
                    st.success(result_text)
                    
                    # Add visualization tabs
                    viz_tab1, viz_tab2, viz_tab3 = st.tabs(["Loss Curve", "Accuracy", "Confusion Matrix"])
                    
                    with viz_tab1:
                        model.plot_loss_curve()
                    
                    with viz_tab2:
                        model.plot_accuracy_curve(X_val, y_val)
                    
                    with viz_tab3:
                        model.plot_confusion_matrix(X_val, y_val)
                    
                    if len(features) >= 2:
                        fig = plot_decision_boundary(model, X_train, y_train, features)
                        st.pyplot(fig)
        
        with col_kernel3:
            st.markdown("#### 🧠 Sigmoid Kernel SVM")
            st.caption("Uses hyperbolic tangent kernel (similar to neural networks)")
            
            C_sigmoid = st.slider("Regularization (C)", 0.01, 10.0, 1.0, key='sigmoid_C_slider')
            gamma_sigmoid = st.slider("Gamma", 0.001, 1.0, 0.1, key='sigmoid_gamma_slider')
            coef0 = st.slider("Coefficient 0", -1.0, 1.0, 0.0, key='sigmoid_coef0')
            
            # Updated optimizer selection
            optimizer_sigmoid = st.selectbox("Optimization Method", 
                                           ["Quadratic Programming", "Gradient Descent", "Subgradient Descent"], 
                                           key='sigmoid_optimizer')
            
            if optimizer_sigmoid != "Quadratic Programming":
                lr_sigmoid = st.slider("Learning Rate", 0.001, 0.1, 0.01, key='sigmoid_lr')
                max_iter_sigmoid = st.slider("Max Iterations", 100, 5000, 1000, key='sigmoid_max_iter')
                tol_sigmoid = st.slider("Convergence Tolerance", 0.0001, 0.01, 0.001, key='sigmoid_tol')
            
            if st.button("Train Sigmoid Kernel SVM", key='train_sigmoid'):
                with st.spinner("Training in progress..."):
                    start_time = time.time()
                    
                    opt_map = {
                        "Quadratic Programming": "qp",
                        "Gradient Descent": "sgd",
                        "Subgradient Descent": "subgrad"
                    }
                    
                    model = KernelSVM(
                        kernel='sigmoid', 
                        C=C_sigmoid, 
                        gamma=gamma_sigmoid, 
                        coef0=coef0,
                        optimizer=opt_map[optimizer_sigmoid],
                        learning_rate=lr_sigmoid if optimizer_sigmoid != "Quadratic Programming" else 0.01,
                        max_iter=max_iter_sigmoid if optimizer_sigmoid != "Quadratic Programming" else 1000,
                        tol=tol_sigmoid if optimizer_sigmoid != "Quadratic Programming" else 1e-3
                    )
                    model.fit(X_train, y_train)
                    
                    train_acc = accuracy_score(y_train, model.predict(X_train))
                    val_acc = accuracy_score(y_val, model.predict(X_val))
                    training_time = time.time() - start_time
                    
                    result_text = f"""
                    Training Results:
                    - Training Accuracy: {train_acc:.4f}
                    - Validation Accuracy: {val_acc:.4f}
                    - Training Time: {training_time:.2f} seconds
                    - Number of Support Vectors: {len(model.support_vectors)}
                    - Optimization Method: {optimizer_sigmoid}
                    """
                    
                    st.success(result_text)
                    
                    # Add visualization tabs
                    viz_tab1, viz_tab2, viz_tab3 = st.tabs(["Loss Curve", "Accuracy", "Confusion Matrix"])
                    
                    with viz_tab1:
                        model.plot_loss_curve()
                    
                    with viz_tab2:
                        model.plot_accuracy_curve(X_val, y_val)
                    
                    with viz_tab3:
                        model.plot_confusion_matrix(X_val, y_val)
                    
                    if len(features) >= 2:
                        fig = plot_decision_boundary(model, X_train, y_train, features)
                        st.pyplot(fig)

    # =============================================
    # MODEL COMPARISON TAB
    # =============================================
    with tab3:
        st.subheader("Model Comparison")
        
        # Add optimization method comparison section
        st.markdown("### Optimization Method Analysis")
        st.write("Compare different optimization approaches for the same kernel")
        
        kernel_for_comparison = st.selectbox("Select kernel for comparison", 
                                           ['rbf', 'poly', 'sigmoid', 'linear'],
                                           key='kernel_compare')
        C_for_comparison = st.slider("Select C value for comparison", 
                                    0.01, 10.0, 1.0, 
                                    key='C_compare')
        
        if st.button("Compare Optimization Methods"):
            with st.spinner("Running comparison..."):
                analyzer = SVMAnalyzer(X_train, y_train, X_val, y_val)
                analyzer.compare_methods(methods=['qp', 'subgrad'],  # Explicitly pass the methods
                                      kernel=kernel_for_comparison,
                                      C=C_for_comparison)
                
                st.success("Comparison complete!")
                analyzer.plot_comparison()
                analyzer.generate_report()
        
        # Model comparison section
        st.markdown("### Model Type Comparison")
        st.write("Compare different SVM implementations")
        
        if st.button("Train and Compare All Models"):
            models = {
                "🔒 Hard Margin SVM": {
                    'class': HardMarginSVM,
                    'params': {}
                },
                "🧈 Soft Margin SVM": {
                    'class': SoftMarginSVM,
                    'params': {'C': 1.0, 'optimization': 'qp'}
                },
                "🌐 RBF Kernel SVM": {
                    'class': KernelSVM,
                    'params': {'kernel': 'rbf', 'gamma': 0.1, 'C': 1.0}
                },
                "🔺 Polynomial Kernel SVM": {
                    'class': KernelSVM,
                    'params': {'kernel': 'poly', 'degree': 3, 'C': 1.0}
                },
                "🧠 Sigmoid Kernel SVM": {
                    'class': KernelSVM,
                    'params': {'kernel': 'sigmoid', 'gamma': 0.1, 'C': 1.0}
                }
            }
            
            results = []
            
            for name, config in models.items():
                with st.spinner(f"Training {name}..."):
                    start_time = time.time()
                    
                    # Initialize model
                    model = config['class'](**config['params'])
                    model.fit(X_train, y_train)
                    
                    # Calculate metrics
                    train_acc = accuracy_score(y_train, model.predict(X_train))
                    val_acc = accuracy_score(y_val, model.predict(X_val))
                    training_time = time.time() - start_time
                    
                    # Get number of support vectors
                    if hasattr(model, 'support_vectors'):
                        num_sv = len(model.support_vectors)
                    elif hasattr(model, 'alpha'):
                        num_sv = np.sum(model.alpha > 1e-5)
                    else:
                        num_sv = 'N/A'
                    
                    results.append({
                        'Model': name,
                        'Train Accuracy': train_acc,
                        'Validation Accuracy': val_acc,
                        'Num Support Vectors': num_sv,
                        'Training Time (s)': training_time
                    })
                    
                    # Plot decision boundary for each model
                    if len(features) >= 2:
                        st.subheader(f"{name} Decision Boundary")
                        fig = plot_decision_boundary(model, X_train, y_train, features)
                        st.pyplot(fig)
            
            # Display comparison results
            st.subheader("Model Comparison Results")
            results_df = pd.DataFrame(results)
            st.dataframe(results_df.style.format({
                'Train Accuracy': '{:.4f}',
                'Validation Accuracy': '{:.4f}',
                'Training Time (s)': '{:.2f}'
            }).highlight_max(axis=0, subset=['Validation Accuracy']))
            
            # Get best model
            best_model = results_df.loc[results_df['Validation Accuracy'].idxmax()]
            
            st.success(f"""
            **🏆 Best Model Recommendation:**
            - **Model**: {best_model['Model']}
            - **Validation Accuracy**: {best_model['Validation Accuracy']:.4f}
            - **Number of Support Vectors**: {best_model['Num Support Vectors']}
            - **Training Time**: {best_model['Training Time (s)']:.2f} seconds
            """)
            
            # Plot comparison chart
            fig, ax = plt.subplots(figsize=(10, 6))
            x = np.arange(len(results_df))
            width = 0.35
            
            ax.bar(x - width/2, results_df['Train Accuracy'], width, label='Train Accuracy')
            ax.bar(x + width/2, results_df['Validation Accuracy'], width, label='Validation Accuracy')
            
            ax.set_xticks(x)
            ax.set_xticklabels(results_df['Model'], rotation=45)
            ax.set_ylabel("Accuracy")
            ax.set_title("Model Comparison")
            ax.legend()
            
            st.pyplot(fig)

    # Add documentation section
    st.header("📚 Model Documentation")
    with st.expander("Understanding the Different SVM Models"):
        st.markdown("""
        ### 🔒 Hard Margin Linear SVM
        - *Purpose:* Maximizes the margin with strict classification (no misclassification allowed)
        - *Best for:* Perfectly separable data
        - *Pros:* Simple, theoretically optimal for separable data
        - *Cons:* Fails when data isn't perfectly separable
        
        ### 🧈 Soft Margin Linear SVM
        - *Purpose:* Allows some misclassification for better generalization
        - *Best for:* Noisy or slightly non-separable data
        - *Pros:* More robust to noise and outliers
        - *Cons:* Requires tuning of C parameter
        
        ### 🌐 RBF Kernel SVM
        - *Purpose:* Handles non-linear boundaries using Gaussian functions
        - *Best for:* Complex, non-linear decision boundaries
        - *Pros:* Powerful for non-linear problems
        - *Cons:* Requires tuning of gamma parameter
        
        ### 🔺 Polynomial Kernel SVM
        - *Purpose:* Captures polynomial relationships between features
        - *Best for:* Structured, polynomial patterns
        - *Pros:* Can model curved decision boundaries
        - *Cons:* Computationally expensive for high degrees
        
        ### 🧠 Sigmoid Kernel SVM
        - *Purpose:* Uses hyperbolic tangent kernel (similar to neural networks)
        - *Best for:* When data resembles neural network activation patterns
        - *Pros:* Can model complex non-linear relationships
        - *Cons:* May not satisfy Mercer's condition
        """) 
        st.markdown("### Created by :")
        st.image(r"C:\Users\ranee\Dropbox\PC\Desktop\convex\raneen Logo.png", width=400)

if __name__ == "__main__":
    main()