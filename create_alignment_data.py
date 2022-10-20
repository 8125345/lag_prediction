import glob
import os
import numpy as np
from shutil import copyfile
import librosa
import math
import soundfile as sf

import audio_utils
from deepiano.wav2mid import audio_label_data_utils
from deepiano.wav2mid.data import wav_to_spec, hparams_frames_per_second
from deepiano.wav2mid import configs
from deepiano.music import audio_io
from deepiano.music import midi_io
from deepiano.music import sequences_lib
from deepiano.wav2mid import constants

config = configs.CONFIG_MAP['onsets_frames']
split_length = 320

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


def get_wav_copy(base_path, base_dst_dir):
    dst_dir_list = []
    splited_wav_base = base_path + '/' + 'split_result_0.5_rename'
    assert os.path.exists(splited_wav_base)
    equipment_ID_list = glob.glob(os.path.join(splited_wav_base, '*'))
    for equipment_ID in equipment_ID_list:
        if os.path.isfile(equipment_ID):
            continue
        equipment = equipment_ID.split('/')[-1]
        dst_dir = base_dst_dir + '/' + equipment + '/' + 'total'
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        dst_dir_list.append(dst_dir)

        wav_list = glob.glob(os.path.join(equipment_ID, '*/*.wav'))
        for wav in sorted(wav_list):
            wav_name = os.path.split(wav)[1]
            # copyfile(wav, os.path.join(dst_dir, wav_name))
    print(dst_dir_list)
    return dst_dir_list


def get_midi_copy(base_path, dst_dir_list):
    midi_onset_base = base_path + '/' + 'xml_wav_onset'
    assert os.path.exists(midi_onset_base)
    tone_repo_list = glob.glob(os.path.join(midi_onset_base, '*'))
    for dst_dir in sorted(dst_dir_list):
        for tone_repo in sorted(tone_repo_list):
            tone_repo_ID = tone_repo.split('/')[-1]
            midi_onset_list = glob.glob(os.path.join(tone_repo, '*.mid'))
            for midi_onset in sorted(midi_onset_list):
                midi_file_name = os.path.split(midi_onset)[1]
                out_midi_name = f'{tone_repo_ID}_{midi_file_name}'
                copyfile(midi_onset, os.path.join(dst_dir, out_midi_name))



def split_320_to_npy_single(wav_file, base_npy_dst_dir, min_length, max_length):
    midi_file = wav_file.replace('.wav', '.mid')
    assert os.path.exists(wav_file)
    assert os.path.exists(midi_file)
    x = wav_file.split('/')
    _, f = os.path.split(wav_file)
    fn, _ = os.path.splitext(f)
    save_dir = x[-3] + '/' + x[-2] + '/' + fn
    npy_output_dir = os.path.join(base_npy_dst_dir, save_dir)
    if not os.path.exists(npy_output_dir):
        os.makedirs(npy_output_dir)

    ns = midi_io.midi_file_to_note_sequence(midi_file)
    samples = audio_utils.file2arr(wav_file)
    samples = librosa.util.normalize(samples, norm=np.inf)
    # Add padding to samples if notesequence is longer.
    pad_to_samples = int(math.ceil(ns.total_time * config.hparams.sample_rate))
    padding_needed = pad_to_samples - samples.shape[0]
    # if padding_needed > 5 * config.hparams.sample_rate:
    #     raise ValueError(
    #         'Would have padded {} more than 5 seconds to match note sequence total '
    #         'time. ({} original samples, {} sample rate, {} sample seconds, '
    #         '{} sequence seconds) This likely indicates a problem with the source '
    #         'data.'.format(
    #             npy_output_dir, samples.shape[0], config.hparams.sample_rate,
    #             samples.shape[0] / config.hparams.sample_rate, ns.total_time))
    samples = np.pad(samples, (0, max(0, padding_needed)), 'constant')

    if max_length == min_length:
        splits = np.arange(0, ns.total_time, max_length)
    elif max_length > 0:
        splits = audio_label_data_utils.find_split_points(ns, samples, config.hparams.sample_rate, min_length, max_length)
    else:
        splits = [0, ns.total_time]

    all_wav_data = []
    all_ns_data = []
    cnt = 0
    for start, end in zip(splits[:-1], splits[1:]):
        if end - start < min_length:
            continue

        if start == 0 and end == ns.total_time:
            new_ns = ns
        else:
            new_ns = sequences_lib.extract_subsequence(ns, start, end)
        if start == 0 and end == ns.total_time:
            new_samples = samples
        else:
            # the resampling that happen in crop_wav_data is really slow
            # and we've already done it once, avoid doing it twice
            new_samples = audio_io.crop_samples(samples, config.hparams.sample_rate, start, end - start)

        new_tmp_wav = audio_io.samples_to_wav_data(new_samples, config.hparams.sample_rate)
        spec = wav_to_spec(new_tmp_wav, config.hparams)

        velocity_range = audio_label_data_utils.velocity_range_from_sequence(ns)
        _, _, onsets, _, _ = sequence_to_pianoroll_fn(new_ns, velocity_range, hparams=config.hparams)

        max_spec_y = max(max(spec.shape[0], onsets.shape[0]), 320)
        # print(f"最大帧数\t{max_spec_y}")
        if spec.shape[0] < max_spec_y:
            spec = np.pad(spec, ((0, max_spec_y - spec.shape[0]), (0, 0)), 'constant', constant_values=(-100, -100))
        if onsets.shape[0] < max_spec_y:
            onsets = np.pad(onsets, ((0, max_spec_y - onsets.shape[0]), (0, 0)), 'constant', constant_values=(0, 0))

        for i in range(0, max_spec_y, split_length):
            start = i
            end = i + split_length

            if end > max_spec_y:
                start = max_spec_y - split_length
                end = max_spec_y

            chunk_spec = spec[start:end]
            onsets_label = onsets[start:end]

            res_data = np.concatenate((chunk_spec, onsets_label), axis=1)
            dst_npy_dir = os.path.join(npy_output_dir, '%06d.npy' % cnt)
            np.save(dst_npy_dir, res_data)
            cnt += 1
        new_wav_samples = audio_io.wav_data_to_samples(new_tmp_wav, config.hparams.sample_rate)
        all_wav_data.append(new_wav_samples)
        all_ns_data.append(new_ns)

    for i in range(0, len(all_ns_data)):
        dst_wav_name = os.path.join(npy_output_dir, '%06d.wav' % i)
        wav_data = all_wav_data[i]
        sf.write(dst_wav_name, wav_data, config.hparams.sample_rate)

        dst_mid_name = os.path.join(npy_output_dir, '%06d.mid' % i)
        ns_data = all_ns_data[i]
        midi_io.note_sequence_to_midi_file(ns_data, dst_mid_name)


def split_320_to_npy_batch(base_path, base_npy_dst_dir):
    base_wav_midi_path = base_path + '/' + 'wav_midi_total'
    assert os.path.exists(base_wav_midi_path)
    equipment_ID_list = glob.glob(os.path.join(base_wav_midi_path, '*'))
    for equipment_ID in sorted(equipment_ID_list):
        if os.path.isfile(equipment_ID):
            continue
        wav_list = glob.glob(os.path.join(equipment_ID, '*/*.wav'))
        for wav in sorted(wav_list):
            split_320_to_npy_single(wav, base_npy_dst_dir, min_length=5, max_length=10)


def Patch_data_process(base_dst_dir, base_npy_dst_dir):

    Patch_data = ['xml_SC55_000', 'xml_SC55_016', 'xml_SC55_051', 'xml_SC55_053',
                  'xml_SC55_065', 'xml_SC55_104', 'xml_SC55_154', 'xml_SC55_177',
                  'xml_arachno_000', 'xml_arachno_016', 'xml_arachno_051', 'xml_arachno_053',
                  'xml_arachno_065', 'xml_arachno_104', 'xml_arachno_154', 'xml_arachno_177']
    equipment_ID = ['ipad6SC55录音版', 'ipadpro叶青大厦D座3SC55录音版', 'iphonexs111新录音25SC55录音版']

    for equipment in equipment_ID:
        for patch in Patch_data:
            temp_wav = base_dst_dir + '/' + equipment + '/total/' + patch + '.wav'
            assert os.path.exists(temp_wav)
            split_320_to_npy_single(temp_wav, base_npy_dst_dir, min_length=0, max_length=10)



if __name__ == '__main__':
    base_path = '/deepiano_data/zhaoliang/SC55_data/Alignment_data'
    base_dst_dir = base_path + '/' + 'wav_midi_total'
    base_npy_dst_dir = base_path + '/' + 'split_320_npy'

    # dst_dir_list = get_wav_copy(base_path, base_dst_dir)
    # get_midi_copy(base_path, dst_dir_list)
    # test_wav = '/deepiano_data/zhaoliang/SC55_data/Alignment_data/wav_midi_total/ipad6SC55录音版/total/xml_SC55_000.wav'
    # split_320_to_npy_single(test_wav, base_npy_dst_dir, min_length=0, max_length=10)
    # split_320_to_npy_batch(base_path, base_npy_dst_dir)
    Patch_data_process(base_dst_dir, base_npy_dst_dir)




