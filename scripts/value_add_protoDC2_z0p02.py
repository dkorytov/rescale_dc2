"""
"""
from slidingpercentile import python_sliding_rank
import os
from astropy.table import Table
from halotools.empirical_models import enforce_periodicity_of_box


dirname = "/Users/aphearin/Dropbox/protoDC2"
protoDC2_basename = "protoDC2_snapnum_487.hdf5"
protoDC2_z0 = Table.read(os.path.join(dirname, protoDC2_basename), path='data')


protoDC2_z0['x'] = enforce_periodicity_of_box(protoDC2_z0['x'], 256.)
protoDC2_z0['y'] = enforce_periodicity_of_box(protoDC2_z0['y'], 256.)
protoDC2_z0['z'] = enforce_periodicity_of_box(protoDC2_z0['z'], 256.)

w = 1001
print("...calculating sliding ranks")
satmask = protoDC2_z0['isCentral'] == 0
satranks = python_sliding_rank(protoDC2_z0['totalMassStellar'][satmask],
    protoDC2_z0['totalStarFormationRate'][satmask], w)
cenranks = python_sliding_rank(protoDC2_z0['totalMassStellar'][~satmask],
    protoDC2_z0['totalStarFormationRate'][~satmask], w)

protoDC2_z0['totalStarFormationRate_percentile_fixed_totalMassStellar'] = -1.
protoDC2_z0['totalStarFormationRate_percentile_fixed_totalMassStellar'][satmask] = (1. + satranks[satmask])/float(w + 1)
protoDC2_z0['totalStarFormationRate_percentile_fixed_totalMassStellar'][~satmask] = (1. + cenranks[~satmask])/float(w + 1)


print("...writing value-added catalogs to disk")
protoDC2_z0.write('value_added_protoDC2_snapnum_487.hdf5', path='data', overwrite=True)
