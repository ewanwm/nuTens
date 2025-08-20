
from nuTens import units
from nuTens import tensor
from nuTens import dtype
from nuTens.tensor import Tensor
import math as m

# create the tensor
theta = Tensor.zeros(shape=[1], dtype=dtype.scalar_type.float, device=dtype.device_type.cpu, requires_grad=True)

# set the value
theta.requires_grad(False)
theta.set_value([0], m.pi / 8.0)
theta.requires_grad(True)

# print the tensor
print(theta.to_string())

# get the value as a raw float
theta_value = theta.get_value([0])
print(f"theta as float = {theta_value}")

# now built the sin^2( 2 theta ) term using tensor operations
sin_2_theta = tensor.sin(tensor.scale(theta, 2.0))
sin_squared_2_theta = tensor.pow(sin_2_theta, 2.0)

# for now we will just use floats for the other variables 
# to keep things (relatively) simple

dm2 = 20.0e-4 * units.eV * units.eV
baseline = 295 * units.km
energy = 0.5 * units.GeV

phi = dm2 * baseline / ( 4.0 * energy )

sin_squared_phi = m.sin(phi) * m.sin(phi)

osc_prob = tensor.scale(sin_squared_2_theta, sin_squared_phi)

print(f"oscillation probability = {osc_prob.to_string()}")

# perform the backpropagation
osc_prob.backward()

# get the accumulated gradient in the theta tensor
gradient = theta.grad()

print(f"gradient = {gradient.to_string()}")