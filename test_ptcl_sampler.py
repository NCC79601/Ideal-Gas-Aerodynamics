from utils.ptcl_sampler import ptcl_sampler

ptcl_pos, ptcl_v = ptcl_sampler(10, [100, 100], 300, 28)

print(f'ptcl_pos: {ptcl_pos}')

print(f'ptcl_v: {ptcl_v}')