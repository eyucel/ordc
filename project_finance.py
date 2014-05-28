__author__ = 'Emre'


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



