"""
"""
import os
from slidingpercentile import sliding_percentile
from .load_halos import load_bolshoi_planck_halos
from .load_umachine_mocks import load_umachine_and_value_added_halos


def store_value_added_umachine_bolshoi_catalogs(umachine_fname, bolshoi_halos_fname):
    umachine, bpl_halos = load_umachine_and_value_added_halos(
        umachine_fname, load_bolshoi_planck_halos(bolshoi_halos_fname))

    x = umachine['obs_sm']
    y = umachine['obs_sfr']
    window = 501
    umachine['sfr_percentile_fixed_sm'] = sliding_percentile(x, y, window)

    umachine_basename = os.path.basename(umachine_fname)
    output_umachine_basename = "value_added_" + umachine_basename
    outname_umachine = os.path.join(os.path.dirname(umachine_fname), output_umachine_basename)

    bolshoi_basename = os.path.basename(bolshoi_halos_fname)
    output_bolshoi_basename = "value_added_" + bolshoi_basename
    outname_bolshoi = os.path.join(os.path.dirname(bolshoi_halos_fname), output_bolshoi_basename)

    umachine.write(outname_umachine, path='data')
    bpl_halos.write(outname_bolshoi, path='data')
