import glob
import os
import numpy as np
from shutil import copyfile

from pkl_to_midi import pkl_to_mid
from deepiano.wav2mid import configs
from deepiano.music import midi_io
from deepiano.wav2mid import audio_label_data_utils
from deepiano.music import sequences_lib
from deepiano.wav2mid import constants
from deepiano.wav2mid.data import hparams_frames_per_second

config = configs.CONFIG_MAP['onsets_frames']

need_pad_time = 160 #ms
need_pad_frame = 5

def sequence_to_pianoroll_fn(sequence, velocity_range, hparams):
    """Converts sequence to pianorolls."""
    #sequence = sequences_lib.apply_sustain_control_changes(sequence)
    roll = sequences_lib.sequence_to_pianoroll(
        sequence,
        frames_per_second=hparams_frames_per_second(hparams),
        min_pitch=constants.MIN_MIDI_PITCH,
        max_pitch=constants.MAX_MIDI_PITCH,
        min_frame_occupancy_for_label=hparams.min_frame_occupancy_for_label,
        onset_mode=hparams.onset_mode,
        onset_length_ms=hparams.onset_length,
        offset_length_ms=hparams.offset_length,
        onset_delay_ms=hparams.onset_delay,
        min_velocity=velocity_range.min,
        max_velocity=velocity_range.max)
    return (roll.active, roll.weights, roll.onsets, roll.onset_velocities,
            roll.offsets)

def single_midi_process(midi_file):
    assert os.path.exists(midi_file)
    ns = midi_io.midi_file_to_note_sequence(midi_file)
    velocity_range = audio_label_data_utils.velocity_range_from_sequence(ns)
    _, _, onsets, _, _ = sequence_to_pianoroll_fn(ns, velocity_range, hparams=config.hparams)
    onsets = np.pad(onsets, ((need_pad_frame, 0), (0, 0)), 'constant', constant_values=(0, 0))

    pkl_to_mid.convert_to_midi_single(onsets, midi_file)


def batch_midi_process(base_path):
    assert os.path.exists(base_path)
    equipment_ID_list = glob.glob(os.path.join(base_path, '*'))
    for equipment_ID in sorted(equipment_ID_list):
        if os.path.isfile(equipment_ID):
            continue
        midi_list = glob.glob(os.path.join(equipment_ID, '*/*.mid'))
        for midi_file in sorted(midi_list):
            single_midi_process(midi_file)


if __name__ == '__main__':
    base_path = '/deepiano_data/zhaoliang/SC55_data/Alignment_data/correct_final_total'

    # single_midi_for_test = '/deepiano_data/zhaoliang/SC55_data/Alignment_data/test_for_midi/xml_arachno_000.mid'
    # single_midi_process(single_midi_for_test)
    batch_midi_process(base_path)


