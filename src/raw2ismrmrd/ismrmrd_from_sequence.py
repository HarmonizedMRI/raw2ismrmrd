"""Create ISMRMRD file from a pulse-sequence file and a raw data stream."""

from collections.abc import Sequence
from pathlib import Path

import ismrmrd
import numpy as np
import pypulseq as pp

from raw2ismrmrd.utils import Fov
from raw2ismrmrd.utils import Limits
from raw2ismrmrd.utils import MatrixSize
from raw2ismrmrd.utils import create_header


def ismrmrd_from_sequence(
    adc_data_list: Sequence[np.ndarray], filename_seq: Path | str, filename_mrd: Path | str, replace_mrd: bool = False
) -> ismrmrd.Dataset:
    """Create ismrmrd file based on list of adc data and pulseq sequence file.

    Parameters
    ----------
    adc_data_list
        list of numpy arrays where each array is one acquisition raw data block
    filename_seq
        filename for sequence file
    filename_mrd
        filename for output ISMRMRD file

    Returns
    -------
       ISMRMRD raw data file
    """
    if isinstance(filename_mrd, str):
        filename_mrd = Path(filename_mrd)

    if filename_mrd.exists():
        if replace_mrd:
            filename_mrd.unlink()
        else:
            raise ValueError(f'{filename_mrd} already exists. Please delete file or set "replace_mrd" to True.')

    sequence = pp.Sequence()
    sequence.read(str(filename_seq))

    adc_labels = sequence.evaluate_labels(evolution='adc')
    # Make labels into lists rather than numpy arrays because ismrmrd cannot deal well with numpy
    for key in adc_labels:
        adc_labels[key] = adc_labels[key].tolist()

    if (n_labels := len(adc_labels.get('LIN', 0))) != (n_adc_data := len(adc_data_list)):
        raise ValueError(f'Number of acquisitions ({n_adc_data}) and labels ({n_labels}) do not match.')

    # Get adc blocks
    adc_blocks = [sequence.get_block(be).adc for be in sequence.block_events if sequence.get_block(be).adc is not None]

    readout_oversampling = (
        sequence.get_definition('ReadoutOversamplingFactor')
        if sequence.get_definition('ReadoutOversamplingFactor')
        else 1.0
    )

    # Create new file
    ds = ismrmrd.Dataset(str(filename_mrd), create_if_needed=True)

    n_readout = adc_data_list[0].shape[-1]
    num_channels = adc_data_list[0].shape[-2]
    n_phase_encoding = max(adc_labels.get('LIN', 0)) - min(adc_labels.get('LIN', 0)) + 1
    n_slice_encoding = max(adc_labels.get('PAR', 0)) - min(adc_labels.get('PAR', 0)) + 1
    hdr = create_header(
        traj_type='cartesian',
        encoding_fov=Fov(*sequence.get_definition('FOV').tolist()),
        recon_fov=Fov(*sequence.get_definition('FOV').tolist()),
        encoding_matrix=MatrixSize(n_x=n_readout, n_y=n_phase_encoding, n_z=n_slice_encoding),
        recon_matrix=MatrixSize(n_x=int(n_readout / readout_oversampling), n_y=n_phase_encoding, n_z=n_slice_encoding),
        dwell_time=adc_blocks[0].dwell,
        k1_limits=Limits.from_label_list(adc_labels.get('LIN', (0,))),
        k2_limits=Limits.from_label_list(adc_labels.get('PAR', (0,))),
        slice_limits=Limits.from_label_list(adc_labels.get('SLC', (0,))),
        contrast_limits=Limits.from_label_list(adc_labels.get('ECO', (0,))),
        average_limits=Limits.from_label_list(adc_labels.get('AVG', (0,))),
        repetition_limits=Limits.from_label_list(adc_labels.get('REP', (0,))),
        phase_limits=Limits.from_label_list(adc_labels.get('PHS', (0,))),
        set_limits=Limits.from_label_list(adc_labels.get('SET', (0,))),
        h1_resonance_freq=sequence.system.gamma * sequence.system.B0,
    )

    def get_sequence_definition(sequence, definition_parameter: str, value_scaling: float = 1.0):
        value = sequence.get_definition(definition_parameter)
        if isinstance(value, np.ndarray):
            return [val.item() for val in (value_scaling * value)]
        if isinstance(value, np.generic):
            return [value_scaling * float(value)]
        if len(value) == 0:
            return []
        return value

    # Sequence Information
    seq = ismrmrd.xsd.sequenceParametersType()
    seq.TR = get_sequence_definition(sequence, 'TR', 1e3)
    seq.TE = get_sequence_definition(sequence, 'TE', 1e3)
    seq.TI = get_sequence_definition(sequence, 'TI', 1e3)
    hdr.sequenceParameters = seq

    ds.write_xml_header(hdr.toXML())

    # add acquisitions with trajectory information
    for idx, adc_data in enumerate(adc_data_list):
        acq = ismrmrd.Acquisition()
        acq.resize(n_readout, num_channels)
        acq.data[:] = adc_data

        acq.center_sample = round(n_readout / 2)

        acq.idx.kspace_encode_step_1 = adc_labels.get('LIN')[idx] if 'LIN' in adc_labels else 0
        acq.idx.kspace_encode_step_2 = adc_labels.get('PAR')[idx] if 'PAR' in adc_labels else 0
        acq.idx.slice = adc_labels.get('SLC')[idx] if 'SLC' in adc_labels else 0
        acq.idx.contrast = adc_labels.get('ECO')[idx] if 'ECO' in adc_labels else 0
        acq.idx.repetition = adc_labels.get('REP')[idx] if 'REP' in adc_labels else 0
        acq.idx.phase = adc_labels.get('PHS')[idx] if 'PHS' in adc_labels else 0
        acq.idx.set = adc_labels.get('SET')[idx] if 'SET' in adc_labels else 0

        acq.read_dir = (1.0, 0.0, 0.0)
        acq.phase_dir = (0.0, 1.0, 0.0)
        acq.slice_dir = (0.0, 0.0, 1.0)

        # Flags
        if adc_labels.get('NAV', 0)[idx]:
            acq.setFlag(ismrmrd.ACQ_IS_PHASECORR_DATA)

        if adc_labels.get('NOISE', 0)[idx]:
            acq.setFlag(ismrmrd.ACQ_IS_NOISE_MEASUREMENT)

        ds.append_acquisition(acq)

    ds.close()

    return ds
