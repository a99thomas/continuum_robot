import numpy as np
import sympy as sp


class Continuum_arm:
    def __init__(self, n_stages, effective_diameter, spring_properties):
        self.n_stages = n_stages
        self.k_constants = spring_properties[0]
        self.max_length = spring_properties[1]
        self.min_length = spring_properties[2]
        self.diameter = effective_diameter
        print(self.calculate_curvatures(self.max_length, self.min_length, self.diameter))

    def calculate_curvatures(self, max_length, min_length, diameter):
        k1 = 2*(max_length - min_length)/((diameter/2)*(max_length + 2*min_length))
        r1 = 1/k1
        k2 = 2*(max_length - min_length)/((diameter/2)*(2*max_length + min_length))
        r2 = 1/k2
        return r1,r2, k1, k2

spring_properties = np.array([[2.25],[1],[.375]])
Continuum_arm(1, 1, spring_properties)