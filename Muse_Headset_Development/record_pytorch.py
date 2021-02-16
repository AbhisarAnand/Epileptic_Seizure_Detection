import numpy as np
import pandas as pd
import os
from typing import Union, List, Optional
from pathlib import Path
from pylsl import StreamInlet, resolve_byprop
from sklearn.linear_model import LinearRegression
from time import time, sleep, strftime, gmtime
from .stream import find_muse
from .muse import Muse
from .constants import LSL_SCAN_TIMEOUT, LSL_EEG_CHUNK, LSL_PPG_CHUNK, LSL_ACC_CHUNK, LSL_GYRO_CHUNK
import pandas as pd
import numpy as np
import fastai
import torch

# Records a fixed duration of EEG data from an LSL stream into a CSV file

def record(
    duration: int,
    filename=None,
    dejitter=False,
    data_source="EEG",
    continuous: bool = True,
) -> None:
    chunk_length = LSL_EEG_CHUNK
    if data_source == "PPG":
        chunk_length = LSL_PPG_CHUNK
    if data_source == "ACC":
        chunk_length = LSL_ACC_CHUNK
    if data_source == "GYRO":
        chunk_length = LSL_GYRO_CHUNK

    if not filename:
        filename = os.path.join(os.getcwd(), "%s_recording_%s.csv" %
                                (data_source,
                                 strftime('%Y-%m-%d-%H.%M.%S', gmtime())))
        
    channel_idx = 0
    num_of_data = 178
    existing = pd.DataFrame()
    index = ["X1","X2","X3","X4","X5","X6","X7","X8","X9","X10","X11","X12","X13","X14","X15","X16","X17","X18","X19","X20","X21","X22","X23","X24","X25","X26","X27","X28","X29","X30","X31","X32","X33","X34","X35","X36","X37","X38","X39","X40","X41","X42","X43","X44","X45","X46","X47","X48","X49","X50","X51","X52","X53","X54","X55","X56","X57","X58","X59","X60","X61","X62","X63","X64","X65","X66","X67","X68","X69","X70","X71","X72","X73","X74","X75","X76","X77","X78","X79","X80","X81","X82","X83","X84","X85","X86","X87","X88","X89","X90","X91","X92","X93","X94","X95","X96","X97","X98","X99","X100","X101","X102","X103","X104","X105","X106","X107","X108","X109","X110","X111","X112","X113","X114","X115","X116","X117","X118","X119","X120","X121","X122","X123","X124","X125","X126","X127","X128","X129","X130","X131","X132","X133","X134","X135","X136","X137","X138","X139","X140","X141","X142","X143","X144","X145","X146","X147","X148","X149","X150","X151","X152","X153","X154","X155","X156","X157","X158","X159","X160","X161","X162","X163","X164","X165","X166","X167","X168","X169","X170","X171","X172","X173","X174","X175","X176","X177","X178"]
    model = torch.load("EEG_torch.pt")


    print("Looking for a %s stream..." % (data_source))
    streams = resolve_byprop('type', data_source, timeout=LSL_SCAN_TIMEOUT)

    if len(streams) == 0:
        print("Can't find %s stream." % (data_source))
        return

    print("Started acquiring data.")
    inlet = StreamInlet(streams[0], max_chunklen=chunk_length)
    # eeg_time_correction = inlet.time_correction()

    print("Looking for a Markers stream...")
    marker_streams = resolve_byprop(
        'name', 'Markers', timeout=LSL_SCAN_TIMEOUT)

    if marker_streams:
        inlet_marker = StreamInlet(marker_streams[0])
    else:
        inlet_marker = False
        print("Can't find Markers stream.")

    info = inlet.info()
    description = info.desc()

    Nchan = info.channel_count()

    ch = description.child('channels').first_child()
    ch_names = [ch.child_value('label')]
    for i in range(1, Nchan):
        ch = ch.next_sibling()
        ch_names.append(ch.child_value('label'))

    res = []
    timestamps = []
    markers = []
    t_init = time()
    time_correction = inlet.time_correction()
    last_written_timestamp = None
    print('Start recording at time t=%.3f' % t_init)
    print('Time correction: ', time_correction)
    while (time() - t_init) < duration:
        try:
            data, timestamp = inlet.pull_chunk(
                timeout=1.0, max_samples=chunk_length)

            if timestamp:
                # print("Data: " + str(data)) 
                new_arr = pd.DataFrame(arr)
                combine = [existing, new_arr]
                existing = pd.concat(combine).reset_index(drop=True)
                print(len(existing))
                if len(existing) >= num_of_data:
                    row = existing[0:num_of_data]
                    row = row[channel_idx]
                    row = pd.DataFrame(row)
                    row = row.T
                    row.columns = index
                    existing.drop(existing.index[0:num_of_data], inplace=True)
                    existing = existing.reset_index(drop=True)
                    row = row.iloc[0,:]
                    row, clas, probs = model.predict(row)
                    if clas.int() == 1:
                        print("Seizure Detected")
                    else:
                        print("No Seizure Detected")

                res.append(data)
                timestamps.extend(timestamp)
                tr = time()
            if inlet_marker:
                marker, timestamp = inlet_marker.pull_sample(timeout=0.0)
                if timestamp:
                    markers.append([marker, timestamp])

            # Save every 5s
            if continuous and (last_written_timestamp is None or last_written_timestamp + 5 < timestamps[-1]):
                _save(
                    filename,
                    res,
                    timestamps,
                    time_correction,
                    dejitter,
                    inlet_marker,
                    markers,
                    ch_names,
                    last_written_timestamp=last_written_timestamp,
                )
                last_written_timestamp = timestamps[-1]

        except KeyboardInterrupt:
            break

    time_correction = inlet.time_correction()
    print("Time correction: ", time_correction)

    _save(
        filename,
        res,
        timestamps,
        time_correction,
        dejitter,
        inlet_marker,
        markers,
        ch_names,
    )

    print("Done - wrote file: {}".format(filename))


def _save(
    filename: Union[str, Path],
    res: list,
    timestamps: list,
    time_correction,
    dejitter: bool,
    inlet_marker,
    markers,
    ch_names: List[str],
    last_written_timestamp: Optional[float] = None,
):
    res = np.concatenate(res, axis=0)
    timestamps = np.array(timestamps) + time_correction

    if dejitter:
        y = timestamps
        X = np.atleast_2d(np.arange(0, len(y))).T
        lr = LinearRegression()
        lr.fit(X, y)
        timestamps = lr.predict(X)

    res = np.c_[timestamps, res]
    data = pd.DataFrame(data=res, columns=["timestamps"] + ch_names)

    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)

    if inlet_marker and markers:
        n_markers = len(markers[0][0])
        for ii in range(n_markers):
            data['Marker%d' % ii] = 0
        # process markers:
        for marker in markers:
            # find index of markers
            ix = np.argmin(np.abs(marker[1] - timestamps))
            for ii in range(n_markers):
                data.loc[ix, "Marker%d" % ii] = marker[0][ii]

    # If file doesn't exist, create with headers
    # If it does exist, just append new rows
    if not Path(filename).exists():
        # print("Saving whole file")
        data.to_csv(filename, float_format='%.3f', index=False)
    else:
        # print("Appending file")
        # truncate already written timestamps
        data = data[data['timestamps'] > last_written_timestamp]
        data.to_csv(filename, float_format='%.3f', index=False, mode='a', header=False)



# Rercord directly from a Muse without the use of LSL


def record_direct(duration,
                  address,
                  filename=None,
                  backend='auto',
                  interface=None,
                  name=None):
    if backend == 'bluemuse':
        raise (NotImplementedError(
            'Direct record not supported with BlueMuse backend. Use record after starting stream instead.'
        ))

    if not address:
        found_muse = find_muse(name, backend)
        if not found_muse:
            print('Muse could not be found')
            return
        else:
            address = found_muse['address']
            name = found_muse['name']
        print('Connecting to %s : %s...' % (name if name else 'Muse', address))

    if not filename:
        filename = os.path.join(
            os.getcwd(),
            ("recording_%s.csv" % strftime("%Y-%m-%d-%H.%M.%S", gmtime())))

    eeg_samples = []
    timestamps = []

    def save_eeg(new_samples, new_timestamps):
        eeg_samples.append(new_samples)
        timestamps.append(new_timestamps)

    muse = Muse(address, save_eeg, backend=backend)
    muse.connect()
    muse.start()

    t_init = time()
    print('Start recording at time t=%.3f' % t_init)

    while (time() - t_init) < duration:
        try:
            sleep(1)
        except KeyboardInterrupt:
            break

    muse.stop()
    muse.disconnect()

    timestamps = np.concatenate(timestamps)
    eeg_samples = np.concatenate(eeg_samples, 1).T
    recording = pd.DataFrame(
        data=eeg_samples, columns=['TP9', 'AF7', 'AF8', 'TP10', 'Right AUX'])

    recording['timestamps'] = timestamps

    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)

    recording.to_csv(filename, float_format='%.3f')
    print('Done - wrote file: ' + filename + '.')
