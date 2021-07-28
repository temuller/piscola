#!/usr/bin/env python3

import sys
import piscola
import argparse

description = f"PISCOLA (v{piscola.__version__}) - Type Ia light-curve fitter"
usage = "piscola <sn_name> [options]"

def main(args=None):

    if not args:
        args = sys.argv[1:] if sys.argv[1:] else ["--help"]

    parser = argparse.ArgumentParser(prog='piscola',
                                    usage=usage,
                                    description=description
                                    )
    parser.add_argument("sn_name",
                        help="name of the supernova"
                        )
    parser.add_argument("--data_dir",
                        dest="data_dir",
                        action="store",
                        default='data',
                        help="directory where to find the SN file with the light curve data"
                        )
    parser.add_argument("-s",
                        "--save",
                        dest="save",
                        action="store_true",
                        help="saves the SN object into a pickle file"
                        )
    parser.add_argument("-sd",
                        "--save_dir",
                        dest="save_dir",
                        action="store",
                        help="directory where to save the SN object"
                        )

    # masking
    parser.add_argument("--mask_snr",
                        dest="mask_snr",
                        action="store",
                        help="masks the data given a signal-to-noise threshold"
                        )
    parser.add_argument("--mask_phase",
                        dest="mask_phase",
                        action="store",
                        nargs=2,
                        help="masks the data given a phase range (2 argument values required: min and max phases)"
                        )

    # lc fits
    parser.add_argument("-k1",
                        "--kernel1",
                        dest="kernel1",
                        action="store",
                        default="matern52",
                        help="kernel to be used by the gaussian process fit of the time axis",
                       choices=['matern32', 'matern52', 'squaredexp']
                       )
    parser.add_argument("-k2",
                        "--kernel2",
                        dest="kernel2",
                        action="store",
                        default="matern52",
                        help="kernel to be used by the gaussian process fit of the wavelength axis",
                       choices=['matern32', 'matern52', 'squaredexp']
                       )

    # mangling
    parser.add_argument("--min_phase",
                        dest="min_phase",
                        action="store",
                        default=-15,
                        help="minimum phase of the light curves to be used"
                        )
    parser.add_argument("--max_phase",
                        dest="max_phase",
                        action="store_true",
                        default=30,
                        help="maximum phase of the light curves to be used"
                        )
    parser.add_argument("-mk",
                        "--mangling_kernel",
                        dest="mangling_kernel",
                        action="store",
                        default="squaredexp",
                        help="kernel to be used by the mangling function",
                       choices=['matern32', 'matern52', 'squaredexp']
                       )
    parser.add_argument("--ebv",
                        dest="ebv",
                        action="store",
                        help="E(B-V) value to be used instead of the dust maps"
                        )
    parser.add_argument("-ds",
                        "--dust_scaling",
                        dest="dust_scaling",
                        action="store",
                        default=0.86,
                        type=float,
                        help="scaling of the dust maps",
                        choices=[0.86, 1.0]
                        )
    parser.add_argument("-rl",
                        "--reddening_law",
                        dest="reddening_law",
                        action="store",
                        default="fitzpatrick99",
                        help="dust extinction law",
                        choices=['ccm89', 'odonnell94', 'fitzpatrick99', 'calzetti00', 'fm07']
                        )
    parser.add_argument("-sfd",
                        "--dustmaps_dir",
                        dest="dustmaps_dir",
                        action="store",
                        default=None
                        )

    args = parser.parse_args(args)

    sn = piscola.call_sn(args.sn_name, args.data_dir)

    # apply maks
    if args.mask_snr:
        sn.mask_data()
    if args.mask_phase:
        min_phase, max_phase = float(args.mask_phase[0]), float(args.mask_phase[1])
        sn.mask_data(mask_snr=False, mask_phase=True, min_phase=min_phase, max_phase=max_phase)

    sn.normalize_data()
    sn.fit_lcs(args.kernel1, args.kernel2)
    sn.mangle_sed(args.min_phase, args.max_phase, kernel=args.mangling_kernel, scaling=args.dust_scaling,
                  reddening_law=args.reddening_law, dustmaps_dir=args.dustmaps_dir, ebv=args.ebv)
    sn.calculate_lc_params()

    if args.save:
        sn.save_sn(path=args.save_dir)

if __name__ == "__main__":
    main(sys.argv[1:])
