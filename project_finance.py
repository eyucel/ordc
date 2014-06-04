__author__ = 'Emre'

from collections import namedtuple
import numpy as np
import scipy.stats as stats

class Turbine(object):

    __slots__ = 'name', 'output', 'efficiency', 'heat_rate', 'cost'

    def __init__(self, choice=1):
        if choice == 1:
            self.name = 'General Electric 107H'
            self.output = 400
            self.efficiency = 0.6
            self.heat_rate = 5.687
            self.cost = 552
        elif choice == 2:
            self.name = 'Siemens SCC6-8000H IS'
            self.output = 410
            self.efficiency = 0.6
            self.heat_rate = 5.687
            self.cost = 565
        elif choice == 3:
            self.name = 'Mitsubishi MPCP(M501J)'
            self.output = 470
            self.efficiency = 0.615
            self.heat_rate = 5.548
            self.cost = 540
        else:
            raise Exception('Turbine Selection Error')

years = 20
tothrs = 365*24
on_peak_days = 170
off_peak_days = 365-on_peak_days
on_peak_hrs = on_peak_days * 24
off_peak_hrs = off_peak_days * 24
plant_output = 410

cap_case = 0.75
capacity_factor = np.ones(years) * cap_case
capacity_factor[0:5] = [.668, .734, .768, .782, .75]

total_plant_output = plant_output * capacity_factor * tothrs
heat_rate = 7.1

power_price = np.zeros((years, 3))
pp_esc = 0.03
power_price[:, 0] = np.array([40.54 * (1+pp_esc)**y for y in range(0, years)])
power_price[:, 1] = np.array([28.92 * (1+pp_esc)**y for y in range(0, years)])
power_price[:, 2] = (power_price[:, 0] * on_peak_hrs + power_price[:, 1] * off_peak_hrs) / tothrs


gp_esc = 0.02
gas_price = np.array([4.39 * (1+gp_esc)**y for y in range(0, years)])


merchant_revenue = total_plant_output * power_price[:, 2]

merchant_fuel_cost = total_plant_output * gas_price * heat_rate

vom_rate = 1.5
vom_cost = total_plant_output * vom_rate

emissions_cost = 0
start_costs = 1.1 * 1e6 * np.ones(years)

energy_gross_margin = merchant_revenue - merchant_fuel_cost - vom_cost - emissions_cost - start_costs

print(energy_gross_margin/1e6)

capacity_price = 0

capacity_revenue = np.ones(years) * capacity_price * plant_output * 365
ancillary_srvcs_revenue = 1.1 * 1e6 * np.ones(years)
total_gross_margin = energy_gross_margin + capacity_revenue + ancillary_srvcs_revenue
total_revenue = ancillary_srvcs_revenue + merchant_revenue

sga_par = 0.02
fixed_om_rate = 15 * 1000 # per mw
sga_cost = total_revenue * sga_par
fixed_om_cost = plant_output * fixed_om_rate

prop_tax_rate = 0.02
investment = 300e6

property_tax = investment * prop_tax_rate

total_other_costs = fixed_om_cost + sga_cost + property_tax

ebitda = total_gross_margin - total_other_costs
print(ebitda/1e6)

depreciation_life = 20
depreciation = investment / depreciation_life # straight line depreciation

ebit = ebitda - depreciation
tax_rate = 0.35

taxes = -ebit * tax_rate

tax_carry = np.zeros(years)

tax_carry[0] = taxes[0]


tax_carry[1:] = np.max(tax_carry[0:years-1] + taxes[1:], 0)
nopat = np.zeros(years)
nopat[0] = ebit[0]
nopat[1:] = ebit[1:] + taxes[1:] + tax_carry[1:]

capex = np.zeros(years)
working_cap = np.zeros(years)
free_cash_flow = nopat + depreciation + capex + working_cap

ebitdax = 6
npv_rate = 0.0786

npv_vals = np.zeros(years)
npv_vals[:] = free_cash_flow
npv_vals[-1] += ebitdax * ebitda[-1]

print(nopat/1e6)


npv = np.npv(npv_rate, npv_vals)
print(npv_vals/1e6)
print(npv/1e6)

