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
totdays = 365*7
plant_output = 410

cap_case = 0.75
capacity_factor = np.ones(years) * cap_case
capacity_factor[0:5] = [.668, .734, .768, .782, .75]

total_plant_output = plant_output * capacity_factor * totdays
heat_rate = 7100

power_price =
merchant_revenue = total_plant_output * power_price