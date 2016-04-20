#!/usr/bin/env python

import numpy as np

from molmod import electronvolt, angstrom, amu, bar
from molmod.units import *
from molmod.constants import lightspeed

from tamkin import *


def cleanup_frequencies(freqs):
    newfreqs = []
    invcm = lightspeed/centimeter
    print '    Cleaning %i frequencies' % len(freqs)
    for freq in freqs:
        if freq > 0:
            newfreqs.append(freq)
        else:
            if freq < -40.0*invcm:
                print "    Warning: imaginary freq %.1f 1/cm. I hope this corresponds with a transition state." % (freq/invcm)
            else:
                print "    Warning: imaginary freq %.1f 1/cm. We substitute this with an arbitrary frequency of 50 cm-1" % (freq/invcm)
                newfreqs.append(50.0*invcm)
    return np.array(newfreqs)


def make_pf_gas(mapname, lot_sp):
    print 'LOADING gas phase molecule', mapname
    mol = load_molecule_vasp(
        "%s/freq/POSCAR" % mapname,
        "%s/freq/OUTCAR" % mapname,
        "%s/sp_%s/OUTCAR" % (mapname, lot_sp),
        is_periodic=False)
    nma = NMA(mol, ConstrainExt(1.0e-03))
    print
    return PartFun(nma, [ExtTrans(), ExtRot(1), Vibrations(freq_scaling=1.0, zp_scaling=1.0)])


def make_pf_full(mapname, lot_sp):
    print 'LOADING periodic crystal (full Hessian)', mapname
    mol = load_molecule_vasp(
        "%s/freq/POSCAR" % mapname,
        "%s/freq/OUTCAR" % mapname,
        "%s/sp_%s/OUTCAR" % (mapname, lot_sp),
        is_periodic=True)
    nma = NMA(mol, ConstrainExt(1.0))
    nma.freqs = cleanup_frequencies(nma.freqs)
    print
    return PartFun(nma, [Vibrations(freq_scaling=1.0, zp_scaling=1.0)])


def make_pf_phva(mapname, lot_sp):
    print 'LOADING periodic crystal (partial Hessian)', mapname
    mol = load_molecule_vasp(
        "%s/freq/POSCAR" % mapname,
        "%s/freq/OUTCAR" % mapname,
        "%s/sp_%s/OUTCAR" % (mapname, lot_sp),
        is_periodic=True)
    fixed = load_fixed_vasp("%s/freq/OUTCAR" % mapname)
    nma = NMA(mol, PHVA(fixed))
    nma.freqs = cleanup_frequencies(nma.freqs)
    print
    return PartFun(nma, [Vibrations(freq_scaling=1.0, zp_scaling=1.0)])


def get_pfs_from_mapnames(mapnames, lot_sp):
    pfs = []
    for mapname in mapnames:
        if mapname.startswith('gas_'):
            pfs.append(make_pf_gas(mapname, lot_sp))
        elif mapname.startswith('full_'):
            pfs.append(make_pf_full(mapname, lot_sp))
        elif mapname.startswith('phva_'):
            pfs.append(make_pf_phva(mapname, lot_sp))
        else:
            raise ValueError('Map name should start with gas_, full_ or phva_. Got "%s"' % mapname)
    return pfs


def reaction_analysis(react_mapnames, ts_mapname, prod_mapnames, lot_sp, pressure, temp):
    '''Thermodynamic analysis of chemical reactions

    Parameters
    ----------
    react_names : list of str
                  A list of directories with reactants. The directory must contain
                  a subdirectory freq with the output of a VASP Hessian calculation
    ts_mapname : str
                 Not supported yet. Use None.
    prod_mapnames : list of str
                    A list of directories with reaction products. Same rules apply.
    lot_sp : str or None
             When given, the energy of every state is read from a VSP single point
             calculation in the subdirectory sp_XXX, where XXX is replaced by lot_sp.
    pressure : float
               The pressure in atomic units.
    temp : float
           The temperature in Kelvin.
    '''
    pf_reacts = get_pfs_from_mapnames(react_mapnames, lot_sp)
    pf_prods = get_pfs_from_mapnames(prod_mapnames, lot_sp)
    tm = ThermodynamicModel(pf_reacts, pf_prods)

    print 'THERMO: %s --> %s' % (' + '.join(react_mapnames), ' + '.join(prod_mapnames))
    print '    Zero-point energy difference [kJ/mol]: %12.4f' % (tm.zero_point_energy_difference()/kjmol)
    print '    Free energy difference       [kJ/mol]: %12.4f' % (tm.free_energy_change(temp)/kjmol)
    print '    Enthalpy difference          [kJ/mol]: %12.4f' % (tm.internal_heat_difference(temp)/kjmol)
    print '    Entropic contribution        [kJ/mol]: %12.4f' % (tm.free_energy_change(temp)/kjmol - tm.internal_heat_difference(temp)/kjmol)
    print

    if ts_mapname is not None:
        pf_trans = get_pfs_from_mapnames([ts_mapnames])[0]
        raise NotImplementedError('If you need this to work, get in touch with Toon')
        DeltaS_RtoTS = (pf_trans.entropy(temp)- sum([pf_react.entropy(temp) for pf_react in pf_reacts]))/(joule/mol)
        print "-T.DeltaS_RtoTS (kJ/mol)",-temp*DeltaS_RtoTS/1000.
        km_fwd = KineticModel(pf_reacts, pf_trans, tunneling=None)
        ra_fwd = ReactionAnalysis(km_fwd, 273, 374,10) #, tunneling=wigner)
#        ra_fwd.plot_arrhenius("%s_arrhenius_fwd.png" % (reactionname))

        km_bwd = KineticModel(pf_prods, pf_trans,  tunneling=None)
        ra_bwd = ReactionAnalysis(km_bwd, 273, 374,10) #, tunneling=wigner)
 #       ra_bwd.plot_arrhenius("reactions/%s/arrhenius_bwd.png" % (mapname)) # make the Arrhenius plot

        # write all results to a file.
        ra_fwd.write_to_file("%s_bimol_fwd.txt" % (reactionname)) # summary of the analysis
 #       entropy  =  (pf_trans.entropy(temp)- sum([pf.entropy(temp) for pf in pf_reacts])
 #       enthalpy = (pf_trans.internal_heat(temp)- sum([pf.internal_heat(temp) for pf in pf_reacts])/kjmol
        dict = [
        ra_fwd.A/km_fwd.unit, ra_fwd.Ea/kjmol,km_fwd.rate_constant(temp,do_log=False)/km_fwd.unit, "%s" % km_fwd.unit_name,
        km_fwd.free_energy_change(temp)/kjmol,km_fwd.zero_point_energy_difference()/kjmol,km_fwd.energy_difference()/kjmol,
        (km_fwd.internal_heat_difference(temp)/kjmol),(km_fwd.free_energy_change(temp)/kjmol-km_fwd.internal_heat_difference(temp)/kjmol),
        ra_bwd.A/km_bwd.unit, ra_bwd.Ea/kjmol,km_bwd.rate_constant(temp,do_log=False)/km_bwd.unit, "%s" % km_bwd.unit_name,
        km_bwd.free_energy_change(temp)/kjmol,km_bwd.zero_point_energy_difference()/kjmol,km_bwd.energy_difference()/kjmol,
        (km_bwd.internal_heat_difference(temp)/kjmol),(km_bwd.free_energy_change(temp)/kjmol-km_bwd.internal_heat_difference(temp)/kjmol),
        tm.free_energy_change(temp)/kjmol,tm.internal_heat_difference(temp)/kjmol,tm.free_energy_change(temp)/kjmol-tm.internal_heat_difference(temp)/kjmol
        ]
        print reactionname, dict


def main():
    cases = [
        (["gas_2pentene","phva_emptyO3"], None, ["phva_2-pentyl"], 'revPBE'),
        (["gas_2pentene","phva_emptyO3"], None, ["phva_3-pentyl"], 'revPBE'),
    ]

    pressure = 1*bar
    temp = 323

    for react_mapnames, ts_mapname, prod_mapnames, lot_sp in cases:
        reaction_analysis(react_mapnames, ts_mapname, prod_mapnames, lot_sp, pressure, temp)


if __name__ == '__main__':
    main()
