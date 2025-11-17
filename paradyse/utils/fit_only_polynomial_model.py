import os
import pickle
import numpy as np

class FitPolynomialModel:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), '0731_dna_fitted_model_only_polynomial.pkl')
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

    def encode_parallel_strategy(self, strategy):
        strategies = ['megatron', 'ulysses', 'metp', 'context_parallel']
        encoding = [0] * len(strategies)
        if strategy in strategies:
            encoding[strategies.index(strategy)] = 1
        return encoding

    def preprocess_X(self, L, H, n, strategy, seq_length, batch_size=1):
        features = [H, n, seq_length, batch_size]
        features.extend(self.encode_parallel_strategy(strategy))
        return np.array(features).reshape(1, -1)

    def predict(self, L, H, n, strategy, seq_length):
        if strategy not in self.model:
            raise ValueError(f'No model for strategy={strategy}')
        m = self.model[strategy]
        key = (L, H, n, strategy)
        
        # 使用多项式拟合进行预测
        poly_info = m['poly_dict'].get(key, None)
        if poly_info is None:
            raise ValueError(f'No polynomial model for (L={L}, H={H}, n={n}, strategy={strategy})')
        
        t_coef = poly_info['time_cost']['coef']
        m_coef = poly_info['max_memory_allocated']['coef']
        t_val = np.polyval(t_coef, seq_length)
        m_val = np.polyval(m_coef, seq_length)
        model_type = 'polynomial'
        
        # 根据具体的(L,H,n)参数组合获取OOM阈值
        oom = m['oom_threshold'].get(key, None)
        return t_val, m_val, oom, model_type

    def get_formula(self, L, H, n, strategy):
        if strategy not in self.model:
            raise ValueError(f'No model for strategy={strategy}')
        m = self.model[strategy]
        key = (L, H, n, strategy)
        max_seq_length = m['max_seq_lengths'].get(key, float('inf'))
        oom = m['oom_threshold'].get(key, None)
        poly_info = m['poly_dict'].get(key, None)
        
        def poly_str(coef):
            terms = []
            deg = len(coef) - 1
            for i, c in enumerate(coef):
                power = deg - i
                if abs(c) < 1e-10:
                    continue
                if power == 0:
                    terms.append(f'{c:.8f}')
                elif power == 1:
                    terms.append(f'{c:.8f}*x')
                else:
                    terms.append(f'{c:.8f}*x**{power}')
            return ' + '.join(terms)
        
        return {
            'time_cost': poly_str(poly_info['time_cost']['coef']) if poly_info else None,
            'max_memory_allocated': poly_str(poly_info['max_memory_allocated']['coef']) if poly_info else None,
            'oom_threshold': oom,
            'max_seq_length': max_seq_length,
            'feature_names': m['feature_names'],
            'model_info': {
                'polynomial_range': f"seq_length > 0",
                'aic_time': poly_info['time_cost']['aic'] if poly_info else None,
                'aic_memory': poly_info['max_memory_allocated']['aic'] if poly_info else None
            }
        }

# 示例用法：
# model = FitPolynomialModel('aaai_experiments/time_memory_cost_modeling/0731_dna_fitted_model_only_polynomial.pkl')
# time_cost, max_mem, oom, model_type = model.predict(L=8, H=8192, n=64, strategy='context_parallel', seq_length=11000)
# print(f"预测结果: 时间={time_cost:.4f}s, 内存={max_mem:.2f}MB, OOM={oom}, 使用模型={model_type}")

# time_cost2, max_mem2, oom2, model_type2 = model.predict(L=8, H=8192, n=64, strategy='context_parallel', seq_length=130000)
# print(f"外推预测: 时间={time_cost2:.4f}s, 内存={max_mem2:.2f}MB, OOM={oom2}, 使用模型={model_type2}") 