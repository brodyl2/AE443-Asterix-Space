from dataclasses import dataclass, field
import pymc as pm
import numpy as np
@dataclass
class Risk:
    name: str = field(init=True)
    baseline_likelihood: float = field(init=True)

    # Schedule Risk Parameterers
    schedule_risk_minimum_value: float = field(init=True, default=0.0)
    schedule_risk_maximum_value: float = field(init=True, default=1.0)
    schedule_risk_most_likely_value: float = field(init=True, default=0.5)

    # Cost Risk Parameterers
    cost_risk_minimum_value: float = field(init=True, default=0.0)
    cost_risk_maximum_value: float = field(init=True, default=1.0)
    cost_risk_most_likely_value: float = field(init=True, default=0.5)

    # Technical Risk Parameters
    technical_risk_minimum_values: list[float] = field(init=True, default_factory=lambda: [])
    technical_risk_maximum_values: list[float] = field(init=True, default_factory=lambda: [])
    technical_risk_most_likely_values: list[float] = field(init=True, default_factory=lambda: [])

    def pert2beta(self, distribution_name, a, b, c):
        mu = (a + 4 * b + c) / 6
        sigma = np.sqrt((mu - a) * (c - mu) / 7)

        alpha = (a - mu) * (a * c - a * mu - c * mu + mu**2 + sigma**2)/(sigma**2 * (c - a))
        beta = -(c - mu) * (a * c - a * mu - c * mu + mu**2 + sigma**2)/(sigma**2 * (c - a))

        return pm.Deterministic(f'{self.name} {distribution_name} Impact',
                                pm.Beta(f'{self.name} {distribution_name} Scaled Impact',
                                        alpha=alpha,
                                        beta=beta) * (c - a) + a
                                )

    def cost_distribution(self):
         return self.pert2beta('Cost',
                               self.cost_risk_minimum_value,
                               self.cost_risk_most_likely_value,
                               self.cost_risk_maximum_value)

    def schedule_distribution(self):
        return self.pert2beta('Schedule',
                               self.schedule_risk_minimum_value,
                               self.schedule_risk_most_likely_value,
                               self.schedule_risk_maximum_value)

    def technical_distributions(self):
        t_dists = []
        for ii in range(len(self.technical_risk_minimum_values)):
            t_dists.append(self.pert2beta(f'Technical Parameter {ii}',
                                            self.technical_risk_minimum_values[ii],
                                            self.technical_risk_most_likely_values[ii],
                                            self.technical_risk_maximum_values[ii]))

        return t_dists