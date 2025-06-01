import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

class FatigueFuzzySystem:
    def __init__(self):
        # Входные переменные
        self.perclos = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'perclos')
        self.mar = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'mar')
        self.pitch  = ctrl.Antecedent(np.arange(0, 301, 1), 'pitch')  # 0…300
        self.yaw = ctrl.Antecedent(np.arange(0, 41, 1),  'yaw')    # 0…40
        self.roll = ctrl.Antecedent(np.arange(0, 101, 1), 'roll')   # 0…100

        # Выходные переменные
        self.fatigue_level = ctrl.Consequent(np.arange(0, 101, 1), 'fatigue_level')
        self.risk_category = ctrl.Consequent(np.arange(0, 4, 1), 'risk_category')

        self._setup_membership_functions()
        self._setup_rules()

        self.system = ctrl.ControlSystem(self.rules)
        self.simulator = ctrl.ControlSystemSimulation(self.system)

    def _setup_membership_functions(self):
        # PERCLOS
        self.perclos['low'] = fuzz.gaussmf(self.perclos.universe, 0, 0.1)
        self.perclos['medium'] = fuzz.gaussmf(self.perclos.universe, 0.3, 0.15)
        self.perclos['high'] = fuzz.gaussmf(self.perclos.universe, 0.6, 0.2)

        # MAR
        self.mar['low'] = fuzz.trapmf(self.mar.universe, [0, 0, 0.2, 0.4])
        self.mar['medium'] = fuzz.trapmf(self.mar.universe, [0.3, 0.45, 0.6, 0.8])
        self.mar['high'] = fuzz.trapmf(self.mar.universe, [0.7, 0.85, 1.0, 1.0])

        # Head pose
        for var in [self.pitch, self.yaw, self.roll]:
            var['normal'] = fuzz.gaussmf(var.universe, 0, 5)
            var['moderate'] = fuzz.gaussmf(var.universe, 15, 5)
            var['extreme'] = fuzz.gaussmf(var.universe, 30, 5)

        # Outputs
        self.fatigue_level.automf(3, names=['low', 'medium', 'high'])
        self.risk_category['normal'] = fuzz.trimf(self.risk_category.universe, [0, 0, 1])
        self.risk_category['medium'] = fuzz.trimf(self.risk_category.universe, [0.5, 1.5, 2.5])
        self.risk_category['high'] = fuzz.trimf(self.risk_category.universe, [2, 3, 3])

    def _setup_rules(self):
        self.rules = []
        # Основные правила
        self.rules.append(ctrl.Rule(
            self.perclos['high'] | self.mar['high'],
            (self.fatigue_level['high'], self.risk_category['high'])
        ))
        self.rules.append(ctrl.Rule(
            (self.pitch['extreme'] | self.yaw['extreme'] | self.roll['extreme']) &
            (self.perclos['medium'] | self.mar['medium']),
            (self.fatigue_level['high'], self.risk_category['high'])
        ))
        self.rules.append(ctrl.Rule(
            (self.perclos['low'] & self.mar['low']),
            (self.fatigue_level['low'], self.risk_category['normal'])
        ))
        self.rules.append(ctrl.Rule(
            self.perclos['medium'] & self.mar['medium'],
            (self.fatigue_level['medium'], self.risk_category['medium'])
        ))

    def evaluate(self, metrics):
        # Подготовка
        inputs = {
            'perclos': float(metrics['perclos']),
            'mar': float(metrics['mar']),
            'pitch': float(metrics['pitch']),
            'yaw': float(metrics['yaw']),
            'roll': float(metrics['roll'])
        }
        for key, val in inputs.items():
            lo, hi = getattr(self, key).universe[[0, -1]]
            self.simulator.input[key] = max(lo, min(hi, val))
        self.simulator.compute()
        return {
            'fatigue_level': self.simulator.output['fatigue_level'],
            'risk_category': self.simulator.output['risk_category']
        }