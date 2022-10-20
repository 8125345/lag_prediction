import glob
import os
from shutil import copyfile

from pkl_to_midi import pkl_to_mid
from deepiano.wav2mid import configs
from deepiano.music import midi_io
from deepiano.wav2mid import audio_label_data_utils
from deepiano.music import sequences_lib
from deepiano.wav2mid import constants
from deepiano.wav2mid.data import hparams_frames_per_second

config = configs.CONFIG_MAP['onsets_frames']

def generate_midi_list(base_path):
    assert os.path.isdir(base_path)
    dir_list = glob.glob(os.path.join(base_path, '*'))
    xml_SC55_midi_list = []
    xml_arachno_midi_list = []
    for dir_ in sorted(dir_list):
        if 'xml_SC55' in dir_:
            xml_SC55_midi_list = glob.glob(os.path.join(dir_, '*.mid'))
        if 'xml_arachno' in dir_:
            xml_arachno_midi_list = glob.glob(os.path.join(dir_, '*.mid'))
    return sorted(xml_SC55_midi_list), sorted(xml_arachno_midi_list)


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


def midi_process(midi_file_path, dst_dir):
    assert os.path.isfile(midi_file_path)
    print(midi_file_path)
    ns = midi_io.midi_file_to_note_sequence(midi_file_path)
    velocity_range = audio_label_data_utils.velocity_range_from_sequence(ns)
    _, _, onsets, _, _ = sequence_to_pianoroll_fn(ns, velocity_range, hparams=config.hparams)


    file_dir, file_name = os.path.split(midi_file_path)
    destination_dir = file_dir.replace(base_path, dst_dir)
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
    dst_file = os.path.join(destination_dir, file_name)
    pkl_to_mid.convert_to_midi_single(onsets, dst_file)


if __name__ == '__main__':
    base_path = '/deepiano_data/zhaoliang/xml_wav'
    dst_dir = '/deepiano_data/zhaoliang/xml_wav_onset'
    xml_SC55_midi_list, xml_arachno_midi_list = generate_midi_list(base_path)
    for xml_SC55_midi in xml_SC55_midi_list:
        midi_process(xml_SC55_midi, dst_dir)
    for xml_arachno_midi in xml_arachno_midi_list:
        midi_process(xml_arachno_midi, dst_dir)


