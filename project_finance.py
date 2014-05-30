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


Peak = namedtuple('Peak', ['summer_peak', 'summer_lo_peak', 'summer_off_peak', 'roy', 'super_peak'])

AE_cap = Peak(550, 550, 420, 300, 550)
dispatch_cap = Peak(400, 150, 150, 150, 400)
PPA_price = Peak(40, 40, 40, 40, 40)
PPA_cap = Peak(400, 150, 150, 150, 400)
PPA_escalator = 0
PPTA_price = Peak(12, 12, 12, 12, 12)
PPTA_cap = Peak(550, 550, 550, 550, 550)
spot_price = Peak(55, 42, 13, 13, 9000)


ancillary_srvcs = 550000
ancillary_esc = 0.015

avail_hrs = Peak(918, 1530, 1224, 5088, 0)
avail_hrs_sp = Peak(918, 1530, 1224, 5072, 16)

prop_tax_rate = 0.02

params = {}
params['OM_fixed'] = 15
params['OM_variable'] = 3.54
params['start_cost'] = 9125000
params['sga_expense'] = 0.02

#model parameters
start_year = 2014
end_year = 2050
contract = 'PPA'
gas_option = 1



years = end_year - start_year + 1
gas_price_deck = np.zeros((4,years))
gas_price_deck[0,:] = 3
gas_price_deck[1,:] = 3.5
gas_price_deck[2,:] = 4
gas_price_deck[3,:] = 4.5

gas_price = gas_price_deck[gas_option, :]

# houston inputs
ccgt = Turbine(2)


nox_output = 0.066/2204.6
nox_price = 77000
nox_cost = nox_price * nox_output * 365 * 24 * 1.3 * ccgt.output

sox_output = 0.003/22.046
sox_price = 0
sox_cost = sox_price * sox_output * 365 * 24 * 1.3 * ccgt.output

vocs_output = 0.0069/2204.6
vocs_price = 0
vocs_cost = vocs_price * vocs_output * 365 * 24 * 1.3 * ccgt.output

air_quality_cost = nox_cost
turbine_cost = ccgt.cost * ccgt.output * 1000
epc_cost = 30 * ccgt.output * 1000

permitting_cost = 500000
consulting_cost = 1000000
legal_cost = 500000
development_profit = 50 * ccgt.output * 1000

total_cost = air_quality_cost + permitting_cost + consulting_cost + legal_cost + development_profit + turbine_cost + epc_cost

p = 0.3
super_peak_events = stats.bernoulli.rvs(p, size=years)

hour_sched = np.outer((1-super_peak_events), np.array(avail_hrs)) + np.outer(super_peak_events, np.array(avail_hrs_sp))

dispatch_cap = np.tile(dispatch_cap, (years, 1))
PPA_cap = np.tile(PPA_cap, (years, 1))
PPTA_cap = np.tile(PPTA_cap, (years, 1))
plant_output = dispatch_cap * hour_sched
print(plant_output)
total_plant_output = np.sum(plant_output, 1)


PPA_revenue = PPA_price * PPA_cap * hour_sched
PPTA_revenue = PPTA_price * PPTA_cap * hour_sched
print(PPTA_revenue)
total_PPA_revenue = np.sum(PPA_revenue, 1)
total_PPTA_revenue = np.sum(PPTA_revenue, 1)

spot_price_esc = 0.015
spot_price = np.array([np.array([55, 42, 13, 13, 9000]) * (1+spot_price_esc)**y for y in range(0, years)])


amt_to_grid = np.maximum(dispatch_cap-PPA_cap, 0)

merchant_revenue = spot_price * amt_to_grid * hour_sched
total_merchant_revenue = np.sum(merchant_revenue, 1)
ancillary_srvcs_rev = np.array([ancillary_srvcs * (1+ancillary_esc)**y for y in range(0, years)])
print(ancillary_srvcs_rev)

other_revenue = ancillary_srvcs_rev + 0

total_revenue = total_merchant_revenue + other_revenue + ( total_PPA_revenue if contract == 'PPA' else total_PPTA_revenue)
print(total_revenue)

fuel_consumed = (total_plant_output if contract == 'PPA' else np.sum(amt_to_grid * hour_sched, 1)) * ccgt.heat_rate

print(fuel_consumed) # TODO CHECK THIS FOR SAN ANTONIO CASE
fuel_cost = fuel_consumed *gas_price

replacement_power = np.zeros((years,5))
replacement_power_cost = replacement_power * spot_price * hour_sched
total_replacement_power_cost = np.sum(replacement_power * spot_price * hour_sched, 1)

OM_fixed_cost = np.zeros(years)



property_tax = np.zeros(years)
OM_fixed_cost[:] = params['OM_fixed'] * ccgt.output * 1000
OM_variable_cost = params['OM_variable'] * total_plant_output
start_cost = params['start_cost'] * np.ones(years)
sga_expense = params['sga_expense'] * total_revenue

macrs_sched = np.array([3.75, 7.22, 6.70, 6.20, 5.80, 5.30, 5.00, 4.52, 4.62, 4.62,
                        4.62, 4.62, 4.62, 4.62, 4.62, 4.62, 4.62, 4.62, 4.62, 4.69])

