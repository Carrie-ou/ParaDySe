import os
import pickle
import numpy as np

class FitModel:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), '/workspace/paradyse/aaai_experiments/time_memory_cost_modeling/0728_fitted_model.pkl')
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
        # 检查是否超出训练范围
        max_seq_length = m['max_seq_lengths'].get(key, float('inf'))
        use_polynomial = seq_length > max_seq_length
        if use_polynomial:
            # 使用每个参数组合独立的多项式回归外推
            poly_info = m['poly_dict'].get(key, None)
            if poly_info is None:
                raise ValueError(f'No polynomial model for (L={L}, H={H}, n={n}, strategy={strategy})')
            t_coef = poly_info['time_cost']['coef']
            m_coef = poly_info['max_memory_allocated']['coef']
            t_val = np.polyval(t_coef, seq_length)
            m_val = np.polyval(m_coef, seq_length)
            model_type = 'polynomial'
        else:
            # 使用随机森林进行内插
            X = self.preprocess_X(L, H, n, strategy, seq_length)
            t_val = m['rf_time'].predict(X)[0]
            m_val = m['rf_mem'].predict(X)[0]
            model_type = 'random_forest'
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
                'random_forest_range': f"seq_length <= {max_seq_length}",
                'polynomial_range': f"seq_length > {max_seq_length}"
            }
        }
