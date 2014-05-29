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
            self.heat_rate = 5687
            self.cost = 552
        elif choice == 2:
            self.name = 'Siemens SCC6-8000H IS'
            self.output = 410
            self.efficiency = 0.6
            self.heat_rate = 5687
            self.cost = 565
        elif choice == 3:
            self.name = 'Mitsubishi MPCP(M501J)'
            self.output = 470
            self.efficiency = 0.615
            self.heat_rate = 5548
            self.cost = 540
        else:
            raise Exception('Turbine Selection Error')


Peak = namedtuple('Peak', ['summer_peak', 'summer_lo_peak', 'summer_off_peak', 'roy', 'super_peak'])

AE_cap = Peak(550, 550, 420, 300, 550)
dispatch_cap = Peak(400, 150, 150, 150, 400)
PPA_price = Peak(40, 40, 40, 40, 40)
PPA_cap = Peak(400, 150, 150, 150, 400)
ppa_escalator = 0
PPTA_price = Peak(12, 12, 12, 12, 12)
PPTA_cap = Peak(550, 550, 550, 550, 550)
spot_price = Peak(55, 42, 13, 13, 9000)


ancillary_srvcs = 550000
ancillary_esc = 0.015

avail_hrs = Peak(918, 1530, 1224, 5088, 0)
avail_hrs_sp = Peak(918, 1530, 1224, 5072, 16)

prop_tax_rate = 0.02

OM_fixed = 15
OM_variable = 3.54
start_cost = 9125000
sga_expense = 0.02

start_year = 2014
end_year = 2050
years = end_year - start_year + 1
gas_price_deck = np.zeros((4,years))
gas_price_deck[0,:] = 3
gas_price_deck[1,:] = 3.5
gas_price_deck[2,:] = 4
gas_price_deck[3,:] = 4.5

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


dispatch_cap = np.tile(dispatch_cap, (years, 1))
PPA_cap = np.tile(PPA_cap, (years, 1))
plant_output = dispatch_cap * (np.outer((1-super_peak_events), np.array(avail_hrs)) + np.outer(super_peak_events, np.array(avail_hrs_sp)))
print(plant_output)
total_plant_output = np.sum(plant_output, 1)


PPA_revenue =