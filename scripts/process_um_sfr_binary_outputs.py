""" Script to process outputs of UniverseMachine stored in the following
location on edison: /scratch2/scratchdirs/aphearin/UniverseMachine/1213/universemachine/output_mocks
"""
import os
import argparse
import numpy as np
from astropy.table import Table


parser = argparse.ArgumentParser()
parser.add_argument("fname", help="Path of the UniverseMachine sfr_catalog_*.bin file")

dt = np.dtype([('id', '<i8'), ('descid', '<i8'), ('upid', '<i8'),
    ('flags', '<i4'), ('uparent_dist', '<f4'), ('pos', '<f4', (6,)),
    ('vmp', '<f4'), ('lvmp', '<f4'), ('mp', '<f4'), ('m', '<f4'), ('v', '<f4'),
    ('r', '<f4'), ('rank1', '<f4'), ('rank2', '<f4'), ('ra', '<f4'),
    ('rarank', '<f4'), ('t_tdyn', '<f4'), ('sm', '<f4'), ('icl', '<f4'),
    ('sfr', '<f4'), ('obs_sm', '<f4'), ('obs_sfr', '<f4'), ('obs_uv', '<f4'), ('foo', '<f4')])

fname = "/Users/aphearin/work/random/1218/data/sfr_catalog_0.501122.bin"

args = parser.parse_args()

data = np.fromfile(args.fname, dtype=dt)

basename = os.path.basename(args.fname)
dirname = os.path.dirname(args.fname)

outname = os.path.join(dirname, basename[:-4]+'.hdf5')

keys_to_keep = ['id', 'upid', 'pos']
t = Table()
t['id'] = data['id']
t['x'] = data['pos'][:, 0]
t['y'] = data['pos'][:, 1]
t['z'] = data['pos'][:, 2]
t['vx'] = data['pos'][:, 3]
t['vy'] = data['pos'][:, 4]
t['vz'] = data['pos'][:, 5]
t['obs_sm'] = data['obs_sm']
t['obs_sfr'] = data['obs_sfr']
t['mpeak'] = data['mp']
t['mvir'] = data['m']
t['vmax'] = data['v']
t['vmax_at_mpeak'] = data['vmp']
t['upid'] = data['upid']

subhalo_mask = t['upid'] != -1
t['hostid'] = t['id']
t['hostid'][subhalo_mask] = t['upid'][subhalo_mask]

t = t[t['obs_sm'] > 10**9]

t.write(outname, path='data', overwrite=True)
