from .fit_model import FitModel
from .fit_only_polynomial_model import FitPolynomialModel

dna_predicted_model = FitModel('/workspace/paradyse/aaai_experiments/time_memory_cost_modeling/0729_dna_fitted_model.pkl')
github_code_predicted_model = FitModel('/workspace/paradyse/aaai_experiments/time_memory_cost_modeling/0729_github_code_fitted_model.pkl')
dna_polynomial_predicted_model = FitPolynomialModel('/workspace/paradyse/aaai_experiments/time_memory_cost_modeling/0731_dna_fitted_model_only_polynomial.pkl')

__all__ = [
    "dna_predicted_model",
    "github_code_predicted_model",
    "dna_polynomial_predicted_model"
]
