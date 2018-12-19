"""Prepare Widerface dataset"""
import os
import shutil
import argparse
import zipfile
from gluoncv.utils import makedirs
from mxnet.gluon.utils import check_sha1

_TARGET_DIR = os.path.join('widerface')

def parse_args():
    parser = argparse.ArgumentParser(
        description='Prepare Widerface dataset.',
        epilog='Example: python tools/prepare.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--download-dir', default=None, help='dataset directory on disk')
    args = parser.parse_args()
    return args

def download_wider(path, overwrite=False):
    _CITY_DOWNLOAD_URLS = [
        ('WIDER_train.zip', 'ea80d8614a81ffaf8b3830a2a6807676ca666846'),
        ('WIDER_val.zip', '3643b3045a491b402b46a22e5ccfe1fdcf3d6c68'),
        ('wider_face_split.zip', 'd4949bbb444f2852e84373b0390f6ba6241be931'),
        ('eval_tools.zip', 'bcb6abdc19dac0f853f75b5d03396d5120aef3dc'),
        # ('WIDER_test.zip', 'f7fa64455c1262150b0dc75985b03a94bf655d92'),
        # ('Submission_example.zip', 'eb124c3a3e90ea03cbc60c28b189ba632dc95444'),
    ]
    download_dir = os.path.join(path, 'downloads')
    makedirs(download_dir)
    for filename, checksum in _CITY_DOWNLOAD_URLS:
        filename = os.path.join(download_dir, filename)
        if not check_sha1(filename, checksum):
            raise UserWarning('File {} is downloaded but the content hash does not match. ' \
                              'The repo may be outdated or download may be incomplete. ' \
                              'If the "repo_url" is overridden, consider switching to ' \
                              'the default repo.'.format(filename))
        # extract
        with zipfile.ZipFile(filename, "r") as zip_ref:
            zip_ref.extractall(path=path)
        print("Extracted", filename)

if __name__ == '__main__':
    args = parse_args()
    if args.download_dir is not None:
        if os.path.isdir(_TARGET_DIR):
            os.remove(_TARGET_DIR)
        # make symlink
        os.symlink(args.download_dir, _TARGET_DIR)
    else:
        download_wider(_TARGET_DIR, overwrite=False)
