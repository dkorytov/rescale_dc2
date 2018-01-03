"""
"""
import os
import numpy as np
from astropy.table import Table
import fnmatch
from .value_add_umachine_catalogs import apply_pbcs, add_host_keys, add_ssfr
from colossus.cosmology import cosmology


cosmo = cosmology.setCosmology('planck15')
upenn_datadir = r'/Users/aphearin/work/sdss/meert15'
umachine_dropbox_dirname = "/Users/aphearin/Dropbox/protoDC2/umachine"


def fname_generator(root_dirname, basename_filepat):
    """ Yield the absolute path of all files in the directory tree of ``root_dirname``
    with a basename matching the input pattern
    """

    for path, dirlist, filelist in os.walk(root_dirname):
        for filename in fnmatch.filter(filelist, basename_filepat):
            yield os.path.join(path, filename)


def _parse_scale_factor_from_umachine_sfr_catalog_fname(fname):
    """
    """
    basename = os.path.basename(fname)
    ifirst = len(basename) - basename[::-1].find('_')
    ilast = len(basename) - basename[::-1].find('.') - 1
    return float(basename[ifirst:ilast])


def find_closest_available_umachine_snapshot(z, dirname=umachine_dropbox_dirname):
    """
    """
    available_fnames = list(fname_generator(dirname, 'sfr_catalog*.hdf5'))

    f = _parse_scale_factor_from_umachine_sfr_catalog_fname
    available_snaps = np.array([1./f(fname) - 1. for fname in available_fnames])

    return available_fnames[np.argmin(np.abs(z - available_snaps))]


def load_closest_available_umachine_catalog(z, dirname=umachine_dropbox_dirname):
    """
    """
    fname = find_closest_available_umachine_snapshot(z, dirname=dirname)
    return add_ssfr(apply_pbcs(add_host_keys(Table.read(fname, path='data'))))


def read_meert_catalog(datadir=upenn_datadir, phot_type=4):
    """ Load the Meert et al. 2015 catalog from the collection of .fits files on disk

    This catalog provides improved photometric measurements for galaxies in the
    SDSS DR7 main galaxy sample.

    Parameters
    ----------
    datadir : string, optional
        Path where the collection of .fits files are stored.
        Default value is set at the top of this module.

    phot_type : int, optional
        integer corresponding to the photometry model fit type from the catalog:
        1=best fit, 2=deVaucouleurs, 3=Sersic, 4=DeVExp, 5=SerExp. Default is 4.

    Returns
    -------
    catalogs : FITS records
        Tuple of 7 .fits records

    """
    from astropy.io import fits as pyfits

    if (phot_type < 1) or (phot_type > 5):
        raise Exception('unsupported type of Meert et al. photometry: %d, choose number between 1 and 5')

    datameertnonpar = os.path.join(datadir, 'UPenn_PhotDec_nonParam_rband.fits')
    datameertnonparg = os.path.join(datadir, 'UPenn_PhotDec_nonParam_gband.fits')
    datameert = os.path.join(datadir, 'UPenn_PhotDec_Models_rband.fits')
    datasdss = os.path.join(datadir, 'UPenn_PhotDec_CAST.fits')
    datasdssmodels = os.path.join(datadir, 'UPenn_PhotDec_CASTmodels.fits')
    datameertg = os.path.join(datadir, 'UPenn_PhotDec_Models_gband.fits')
    #  morphology probabilities from Huertas-Company et al. 2011
    datamorph = os.path.join(datadir, 'UPenn_PhotDec_H2011.fits')

    # mdata tables: 1=best fit, 2=deVaucouleurs, 3=Sersic, 4=DeVExp, 5=SerExp
    mdata = pyfits.open(datameert)[phot_type].data
    mdatag = pyfits.open(datameertg)[phot_type].data
    mnpdata = pyfits.open(datameertnonpar)[1].data
    mnpdatag = pyfits.open(datameertnonparg)[1].data
    sdata = pyfits.open(datasdss)[1].data
    phot_r = pyfits.open(datasdssmodels)[1].data
    morph = pyfits.open(datamorph)[1].data

    # eliminate galaxies with bad photometry
    fflag = mdata['finalflag']
    # print("# galaxies in initial Meert et al. sample = {0}".format(np.size(fflag)))

    def isset(flag, bit):
        """Return True if the specified bit is set in the given bit mask"""
        return (flag & (1 << bit)) != 0

    # use minimal quality cuts and flags recommended by Alan Meert
    igood = [(phot_r['petroMag'] > 0.) & (phot_r['petroMag'] < 100.) & (mnpdata['kcorr'] > 0) &
             (mdata['m_tot'] > 0) & (mdata['m_tot'] < 100) &
             (isset(fflag, 1) | isset(fflag, 4) | isset(fflag, 10) | isset(fflag, 14))]

    sdata = sdata[igood]
    phot_r = phot_r[igood]
    mdata = mdata[igood]
    mnpdata = mnpdata[igood]
    mdatag = mdatag[igood]
    mnpdatag = mnpdatag[igood]
    morph = morph[igood]

    t = Table()
    t['objid'] = sdata['objid']
    t['z'] = sdata['z']
    t['ra'] = sdata['ra']
    t['dec'] = sdata['dec']
    t['kcorr'] = mnpdata['kcorr']

    # Calculate bulge-to-total light in r-band
    magr_bulge = mdata['m_bulge']
    magr_disk = mdata['m_disk']
    extmr = mnpdatag['extinction']
    lum_dist = cosmo.luminosityDistance(t['z'])/cosmo.h
    Magr_bulge = (magr_bulge - 5.0*np.log10(lum_dist/1e-5) -
            extmr + 1.3*t['z'] - t['kcorr'])
    Magr_disk = (magr_disk - 5.0*np.log10(lum_dist/1e-5) -
            extmr + 1.3*t['z'] - t['kcorr'])
    bulge_term = 10.**(-Magr_bulge/2.5)
    disk_term = 10.**(-Magr_disk/2.5)
    t['bulge_to_total_rband'] = bulge_term/(bulge_term+disk_term)

    # component half-light radii in kpc
    ang_dist = lum_dist / (1.+t['z'])**2
    t['r50_magr_kpc'] = mdata['r_tot'] * np.pi * ang_dist * 1000.0/(180.*3600.)
    t['r50_magr_disk_kpc'] = mdata['r_disk'] * np.pi * ang_dist * 1000.0/(180.*3600.)
    t['r50_magr_bulge_kpc'] = mdata['r_bulge'] * np.pi * ang_dist * 1000.0/(180.*3600.)

    t.sort('objid')
    unique_objids, indices = np.unique(t['objid'], return_index=True)
    return t[indices]







