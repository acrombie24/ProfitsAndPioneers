import numpy as np
import pandas as pd

'''
Get Patient Market as a Function of US Population Growth Rate and Contraction Rate

INPUTS
    us_population: US population (2023) in millions
    us_population_growth_rate: US population growth rate expressed as a decimal
    n_years: number of years used to calculate NPV
    get_flu_rate: rate of flu contraction in the US expressed as a decimal
    max_penetration: maximum US market penetration expressed as a decimal
    patent_exclusive_years: number of years the drug patent is effective
    penetration_upon_patent_loss: market penetration after the patent expires expressed as a decimal
    inflection_year: number of years after approval where market penetration grows the fastest
    hill_coefficient: steepness parameter for the Fisher-Pry Model
    lead_time_years: remaing number of years before the drug is approved
    rd_cost_scaler: a scaling factor for the R&D costs
    clindev_sga: SG&A costs during clinical development in millions USD
    cost_of_revenue: costs associated with revenue expressed as a decimal
    cost_per_unit: cost of one dose of the drug in USD
    discount_rate: discount rate expressed as a decimal

OUTPUT:
    NPV (in millions USD)
'''

def get_npv(
    us_population: float = 335.5,
    us_population_growth_rate: float = 0.007, 
    n_years: int = 30,
    get_flu_rate: float = 0.14,
    max_penetration: float = 0.5,
    patent_exclusive_years: int = 19,
    penetration_upon_patent_loss: float = 0.05, 
    inflection_year: float = 1.5,
    hill_coefficient: float = 7,
    lead_time_years: int = 6, 
    rd_cost_scaler: float = 1,  
    clindev_sga: float = 0.5, 
    cost_of_revenue: float = 0.49 ,
    cost_per_unit: float = 45,
    discount_rate: float = 0.35
):

    # Market Addressable
    population = us_population * np.array([(1 + us_population_growth_rate) ** i for i in np.arange(0, n_years)])
    n_flu = get_flu_rate * population

    # Market Penetration Rate (Fraction of Flu Getters Using)
    pre_approval = np.zeros(lead_time_years)
    patent_effective = np.array([max_penetration / (1 + np.exp(-1 * hill_coefficient * (i - inflection_year))) for i in np.arange(0, patent_exclusive_years)])
    post_patent = penetration_upon_patent_loss * np.ones(5)
    market_penetration = np.concatenate([pre_approval, patent_effective, post_patent])

    # Number of Patients per Year
    n_patients_treated = market_penetration * n_flu

    # Non-revenue costs
    rd_costs_baseline = np.array([3.515, 3.789, 8.2, 10.8, 23.796, 2.2])
    rd_costs = rd_cost_scaler * rd_costs_baseline
    dev_costs = clindev_sga + rd_costs
    nr_costs = np.concatenate([dev_costs, np.zeros(24)])

    # Revenue
    revenue = n_patients_treated * cost_per_unit

    # Revenue-associated costs
    r_costs = revenue * cost_of_revenue

    # Total costs
    costs = nr_costs + r_costs

    # Net Income
    net_income = revenue - costs

    # DCFs
    dcfs = np.array([x / ((1 + discount_rate) ** t) for x, t in zip(net_income, np.arange(0, n_years))])

    return dcfs.sum()