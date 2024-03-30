from dataclasses import dataclass, field

from pytensor.tensor.variable import TensorVariable
from ISST import Risk, RiskTable

from fpdf import FPDF

import pymc as pm
import numpy as np
import pandas as pd
import arviz as az

import os
from pathlib import Path

from datetime import date
@dataclass
class DesignSystem:
    name: str = field(init=True)

    risks: list[Risk] = field(init=True)

    model_context: pm.Model = field(init=True)

    # System-Wide Schedule Risk Table
    schedule_risk_table: RiskTable = field(init=True)

    # System-Wide Cost Risk Table
    cost_risk_table: RiskTable = field(init=True)

    # System-Wide Technical Risk Tables
    technical_risk_tables: list[RiskTable] = field(init=True)

    def __post_init__(self):
        assert self.schedule_risk_table is not None
        assert self.cost_risk_table is not None
        assert self.technical_risk_tables is not None

        self.schedule_risk_levels = np.zeros(np.asarray(self.schedule_risk_table.utility_breakpoints).shape[0])
        self.cost_risk_levels = np.zeros(np.asarray(self.cost_risk_table.utility_breakpoints).shape[0])

        self.max_tech_risk_sizes = np.zeros((len(self.technical_risk_tables)))
        mtrs = 0
        for ii, risk_table in enumerate(self.technical_risk_tables):
            mtrs = max(mtrs, np.asarray(risk_table.utility_breakpoints).shape[0])
            self.max_tech_risk_sizes[ii] = mtrs

    def generate_system_specification(self):

        rootpath = os.getcwd()
        system_path = Path(rootpath, self.name)
        os.makedirs(system_path, exist_ok=True)

        schedule_df = pd.DataFrame(data={'Minimum Schedule Impact': np.zeros(len(self.risks)),
                                         'Maximum Schedule Impact': np.zeros(len(self.risks)),
                                         'Most Likely Schedule Impact': np.zeros(len(self.risks))},
                                   index=[risk.name for risk in self.risks])

        with open(Path(system_path, f'{self.name} Schedule Risks.csv'), 'w') as f:
            schedule_df.to_csv(f, index=True, header=True)

        cost_df = pd.DataFrame(data={'Minimum Cost Impact': np.zeros(len(self.risks)),
                                     'Maximum Cost Impact': np.zeros(len(self.risks)),
                                     'Most Likely Cost Impact': np.zeros(len(self.risks))},
                               index=[risk.name for risk in self.risks])

        with open(Path(system_path, f'{self.name} Cost Risks.csv'), 'w') as f:
            cost_df.to_csv(f, index=True, header=True)

        for tech_risk in self.technical_risk_tables:
            tech_risk_df = pd.DataFrame(data={f'Minimum {tech_risk.name} Impact': np.zeros(len(self.risks)),
                                              f'Maximum {tech_risk.name} Impact': np.zeros(len(self.risks)),
                                              f'Most Likely {tech_risk.name} Impact': np.zeros(len(self.risks))},
                                        index=[risk.name for risk in self.risks])

            with open(Path(system_path, f'{self.name} {tech_risk.name} Risks.csv'), 'w') as f:
                tech_risk_df.to_csv(f, index=True, header=True)

        return

    def read_system_specification(self):

        rootpath = os.getcwd()
        system_path = Path(rootpath, self.name)

        with open(Path(system_path, f'{self.name} Schedule Risks.csv'), 'r') as f:
            schedule_df = pd.read_csv(f, index_col=0)

            for risk in self.risks:
                risk.schedule_risk_minimum_value = schedule_df.loc[risk.name, 'Minimum Schedule Impact']
                risk.schedule_risk_maximum_value = schedule_df.loc[risk.name, 'Maximum Schedule Impact']
                risk.schedule_risk_most_likely_value = schedule_df.loc[risk.name, 'Most Likely Schedule Impact']

        with open(Path(system_path, f'{self.name} Cost Risks.csv'), 'r') as f:
            cost_df = pd.read_csv(f, index_col=0)

            for risk in self.risks:
                risk.cost_risk_minimum_value = cost_df.loc[risk.name, 'Minimum Cost Impact']
                risk.cost_risk_maximum_value = cost_df.loc[risk.name, 'Maximum Cost Impact']
                risk.cost_risk_most_likely_value = cost_df.loc[risk.name, 'Most Likely Cost Impact']

        for tech_risk in self.technical_risk_tables:

            with open(Path(system_path, f'{self.name} {tech_risk.name} Risks.csv'), 'r') as f:
                tech_risk_df = pd.read_csv(f, index_col=0)

            for risk in self.risks:
                risk.technical_risk_minimum_values.append(
                    tech_risk_df.loc[risk.name, f'Minimum {tech_risk.name} Impact'])
                risk.technical_risk_maximum_values.append(
                    tech_risk_df.loc[risk.name, f'Maximum {tech_risk.name} Impact'])
                risk.technical_risk_most_likely_values.append(
                    tech_risk_df.loc[risk.name, f'Most Likely {tech_risk.name} Impact'])

        return

    def analyze_system(self):
        with self.model_context as model:
            priors              = [risk.baseline_likelihood for risk in self.risks]
            cost_impacts        = [risk.cost_distribution() for risk in self.risks]
            schedule_impacts    = [risk.schedule_distribution() for risk in self.risks]
            tech_impacts        = np.asarray([risk.technical_distributions() for risk in self.risks]).T.tolist()

            TCI = pm.Deterministic('Total Cost Impact', pm.math.dot(priors, cost_impacts))
            TSI = pm.Deterministic('Total Schedule Impact', pm.math.dot(priors, schedule_impacts))
            for ii, tech_risk in enumerate(self.technical_risk_tables):
                TTI = pm.Deterministic(f'Total {tech_risk.name} Impact', pm.math.dot(priors, tech_impacts[ii]))

            return pm.sample()

        return idata




