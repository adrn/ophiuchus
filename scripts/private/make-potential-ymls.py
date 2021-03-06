import sys
import yaml
import astropy.units as u
from gala.units import galactic

alphas = [20,25,30]*u.deg
Omegas = [40,50,60]*u.km/u.s/u.kpc

with open("/Users/adrian/projects/ophiuchus/ophiuchus/potential/yml/barred_mw.yml") as f:
    yaml_pot = yaml.load(f)

comp_ix = None
for i,comp in enumerate(yaml_pot['components']):
    if comp['class'] == 'WangZhaoBarPotential':
        comp_ix = i

if comp_ix is None:
    raise ValueError("Something is horribly wrong")

i = 1
for a in alphas:
    for O in Omegas:
        alpha = a.decompose(galactic).value
        Omega = O.decompose(galactic).value

        yaml_pot['components'][comp_ix]['parameters']['alpha'] = alpha
        yaml_pot['components'][comp_ix]['parameters']['Omega'] = Omega

        fn = "/Users/adrian/projects/ophiuchus/ophiuchus/potential/yml/barred_mw_{}.yml".format(i)
        with open(fn, 'w') as f:
            yaml.dump(yaml_pot, f, default_flow_style=False)

        i += 1

# fixed bar
alpha = (25*u.degree).decompose(galactic).value
Omega = 0.

yaml_pot['components'][comp_ix]['parameters']['alpha'] = alpha
yaml_pot['components'][comp_ix]['parameters']['Omega'] = Omega

fn = "/Users/adrian/projects/ophiuchus/ophiuchus/potential/yml/barred_mw_fixed.yml"
with open(fn, 'w') as f:
    yaml.dump(yaml_pot, f, default_flow_style=False)
