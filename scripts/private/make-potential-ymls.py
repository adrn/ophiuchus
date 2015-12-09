import yaml
import astropy.units as u
from gary.units import galactic

alphas = [20,25,30]*u.deg
Omegas = [40,50,60]*u.km/u.s/u.kpc

with open("/Users/adrian/projects/ophiuchus/ophiuchus/potential/yml/barred_mw.yml") as f:
    yaml_pot = yaml.load(f)

i = 1
for a in alphas:
    for O in Omegas:
        alpha = a.decompose(galactic).value
        Omega = O.decompose(galactic).value

        yaml_pot['parameters']['bar']['alpha'] = alpha
        yaml_pot['parameters']['bar']['Omega'] = Omega

        fn = "/Users/adrian/projects/ophiuchus/ophiuchus/potential/yml/barred_mw_{}.yml".format(i)
        with open(fn, 'w') as f:
            yaml.dump(yaml_pot, f, default_flow_style=False)

        i += 1