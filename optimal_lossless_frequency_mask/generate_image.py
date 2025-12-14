import eca
import numpy
    
def generate_eca_spacetime(rule_number:int=110, width:int=500, height:int=500) -> numpy.ndarray:
    configuration = eca.OneDimensionalElementaryCellularAutomata(lattice_width=width)
    for _ in range(height - 1):
        configuration.transition(rule_number=rule_number)
    return configuration.evolution()