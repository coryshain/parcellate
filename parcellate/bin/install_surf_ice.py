import os
import shutil
import platform
import argparse

from parcellate.util import join, dirname


if __name__ == '__main__':
    argparser = argparse.ArgumentParser('Install Surf Ice for atlas plotting.')
    argparser.add_argument('-o', '--os', type=str, default=None, help='Operating system type (if None, inferred).')
    args = argparser.parse_args()

    os_cur = platform.system().lower()
    if args.os:
        os_name = args.os.lower()
    else:
        os_name = os_cur

    surf_ice_dir = join(dirname(dirname(__file__)), 'resources', 'surfice')
    if not os.path.exists(surf_ice_dir):
        os.makedirs(surf_ice_dir)

    cur_dir = os.getcwd()
    os.chdir(surf_ice_dir)
    if os_name == 'linux':
        os.system('curl -fLO https://github.com/neurolabusc/surf-ice/releases/latest/download/surfice_linux.zip')
        os.system('unzip -o surfice_linux.zip')
    elif os_name == 'darwin':  # Mac
        os.system('curl -fLO https://github.com/neurolabusc/surf-ice/releases/latest/download/surfice_macOS.dmg')
        os.system('7z x surfice_macOS.dmg')
    elif os_name == 'windows':
        os.system('curl -fLO https://github.com/neurolabusc/surf-ice/releases/latest/download/surfice_windows.zip')
        os.system('unzip -o surfice_windows.zip')
        if os_cur == 'linux':  # Installing Windows exe on WSL
            os.system('chmod a+x Surf_Ice/surfice.exe')
    else:
        raise ValueError('OS %s not supported.' % os_name)
    os.chdir(cur_dir)


