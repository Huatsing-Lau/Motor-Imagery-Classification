import pyedflib
import numpy as np
import matplotlib.pyplot as plt
import filter_trial
for sub in range(109):
    real_sub_number = "%03d" % (sub + 1)
    this_sub_address = "S" + str(real_sub_number)
    for run in range(0,14):
        real_run_number = "%02d" % (run + 1)
        this_run_address = "R" + str(real_run_number)

        print("../" + this_sub_address + "/" +
                                      this_sub_address + this_run_address + ".edf")
        edf_data = pyedflib.EdfReader("../" + this_sub_address + "/" +
                                      this_sub_address + this_run_address + ".edf")
        channel_number = edf_data.signals_in_file
        time_number_length = len(edf_data.readSignal(0))
        filter_data_buff = np.zeros([64, time_number_length])
        for channel in range(channel_number):
            time_number = edf_data.readSignal(channel)
            print(len(time_number))
            filter_data_buff[channel, :] = filter_trial.filter_bandpass(time_number)
        np.save(this_sub_address + this_run_address + ".npy", filter_data_buff)
