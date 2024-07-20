'''
This script is used to test the speed of element-wise multiplication of two large arrays.
Does not play any role in the simulation.
'''
import torch
import time

print(f'cuda state: {torch.cuda.is_available()}')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Generate random float arrays a and b
print('generating random float arrays...')
a = torch.rand(int(1e9)).to(device)
b = torch.rand(int(1e9)).to(device)
print('arrays generated')

# Measure the start time
start_time = time.time()

# Compute element-wise multiplication
print(f'computing element-wise multiplication...')
c = torch.mul(a, b)
print(f'computation done')

# Measure the end time
end_time = time.time()

# Calculate the time span in milliseconds
time_span = (end_time - start_time) * 1000

print(f"Elapsed computing time: {time_span} milliseconds")