import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

class RecommendationFuzzySystem:
    def __init__(self):
        # Входы — выходы первого ФИС
        self.fatigue = ctrl.Antecedent(np.arange(0, 101, 1), 'fatigue')
        self.risk = ctrl.Antecedent(np.arange(0, 4, 1), 'risk')
        self.recommendations = ctrl.Consequent(np.arange(0, 5, 1), 'recommendations')

        self._setup_membership()
        self._setup_rules()

        self.system = ctrl.ControlSystem(self.rules)
        self.simulator = ctrl.ControlSystemSimulation(self.system)

    def _setup_membership(self):
        # fatigue: low, medium, high
        self.fatigue['low'] = fuzz.trimf(self.fatigue.universe, [0, 0, 50])
        self.fatigue['medium'] = fuzz.trimf(self.fatigue.universe, [25, 50, 75])
        self.fatigue['high'] = fuzz.trimf(self.fatigue.universe, [50, 100, 100])
        # risk: normal(0), medium(1), high(2+)
        self.risk['normal'] = fuzz.trimf(self.risk.universe, [0, 0, 1])
        self.risk['medium'] = fuzz.trimf(self.risk.universe, [0.5, 1.5, 2.5])
        self.risk['high'] = fuzz.trimf(self.risk.universe, [2, 3, 3])
        # recommendations: none, short_break, physical_activity, medical_check, emergency
        labels = ['none', 'short_break', 'physical_activity', 'medical_check', 'emergency']
        self.recommendations.automf(names=labels)

    def _setup_rules(self):
        self.rules = [
            ctrl.Rule(self.fatigue['low'] & self.risk['normal'], self.recommendations['none']),
            ctrl.Rule(self.fatigue['medium'] | self.risk['medium'], self.recommendations['short_break']),
            ctrl.Rule(self.fatigue['medium'] & self.risk['high'], self.recommendations['physical_activity']),
            ctrl.Rule(self.fatigue['high'] & self.risk['medium'], self.recommendations['medical_check']),
            ctrl.Rule(self.fatigue['high'] & self.risk['high'], self.recommendations['emergency'])
        ]

    def evaluate(self, fatigue_value, risk_value):
        self.simulator.input['fatigue'] = fatigue_value
        self.simulator.input['risk'] = risk_value
        self.simulator.compute()
        return self.simulator.output['recommendations']