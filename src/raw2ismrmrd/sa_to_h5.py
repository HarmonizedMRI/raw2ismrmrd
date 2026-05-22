#!/usr/bin/env python3
"""Convert raw data from a scan archive file to a numpy array."""

import argparse
import logging
from pathlib import Path

import h5py
import numpy as np
from GERecon import Archive

logging.basicConfig(level=logging.INFO)


def get_useful_packets(archive_filename: str) -> int:
    """
    Read a ScanArchive file and count useful opcodes.

    Parameters
    ----------
    archive_filename
        Path to an existing ScanArchive*.h5 file.

    Returns
    -------
        sum where opcode = 1

    """
    archive = Archive(archive_filename)
    metadata = archive.Metadata()
    num_control = metadata['controlCount']
    opcode = []

    # Loop over all the control packets
    for _ in range(num_control):
        # Retrieve the next control packet
        control = archive.NextControl()
        opcode.append(control['opcode'])

    return opcode.count(1)


def read_archive(archive_filename: str) -> None:
    """
    Read a ScanArchive file and convert to a numpy array.

    Parameters
    ----------
    archive_filename
        Path to an existing ScanArchive*.h5 file.

    Returns
    -------
        Numpy array containing the raw data.

    """
    # First figure out how many useful control packets we have got in order to determine the data size
    num_packets = get_useful_packets(archive_filename)
    print(f'Found {num_packets} useful packets')

    # Re-open the archive
    archive = Archive(archive_filename)
    metadata = archive.Metadata()
    print('Metadata: ', metadata)

    num_control = metadata['controlCount']

    output_filename = archive_filename.replace('.h5', '_raw.h5')

    with h5py.File(output_filename, 'w') as h5file:
        # Store metadata in the HDF5 file
        meta_grp = h5file.create_group('metadata')
        for key, value in metadata.items():
            meta_grp.attrs[key] = value

        # Create a group to hold the frames
        frames_grp = h5file.create_group('frames')

        useful_view = 0  # separate counter for opcode==1 frames

        # Loop over all the control packets
        for view in range(num_control):
            # Retrieve the next control packet
            control = archive.NextControl()

            # Raw control packet — consume the frame to stay in sync, but discard it
            if control['opcode'] == 16:
                _ = np.squeeze(archive.NextFrame())

            # Programmable control packet — save the frame to HDF5
            elif control['opcode'] == 1:
                if control['operation'] == 0:
                    frame = np.array(archive.NextFrame())  # ensure it's a numpy array

                    ds = frames_grp.create_dataset(f'frame_{useful_view:06d}', data=frame)

                    # Optionally store per-frame metadata as dataset attributes
                    ds.attrs['view_index'] = view
                    ds.attrs['shape'] = frame.shape

                    useful_view += 1
                else:
                    raise ValueError('Unknown control operation')

        # Store a summary at the top level for easy access later
        h5file.attrs['num_frames_stored'] = useful_view
        h5file.attrs['source_file'] = archive_filename

    print(f'Saved {useful_view} frames to {output_filename}')


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description='Extract raw data from a scan archive file and export as numpy array.')

    parser.add_argument('scan_arc', type=Path, help='Path to Scan Archive')
    args = parser.parse_args()

    scan_arc = args.scan_arc

    # Read ScanArchive using Orchestra Python SDK",
    read_archive(str(scan_arc))
