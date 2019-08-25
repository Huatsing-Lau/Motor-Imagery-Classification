import numpy as np
import pyedflib


def get_dataset():
    seg_buff_all = np.zeros([3600, 64, 640])
    #seg_buff_all = np.zeros([30, 64, 640])
    sample_count = 0
    label_buff_all = np.zeros(3600)
    #label_buff_all = np.zeros(30)
    run_list = ['04', '08', '12']
    for sub in range(80):
        real_sub_number = "%03d" % (sub + 1)
        for run in range(3):
            numpy_data = np.load("S" + str(real_sub_number)
                                 + "R" + str(run_list[run]) + ".npy")
            orginal_data = pyedflib.EdfReader("../S" + str(real_sub_number) + "/S" + str(real_sub_number)
                                              + "R" + str(run_list[run]) + ".edf")
            label_list = orginal_data.readAnnotations()[2]
            seg_length = len(label_list)
            seg_start = orginal_data.readAnnotations()[0]
            seg_start_list = []
            label_list_list = []
            for i in range(seg_length):
                if label_list[i] != 'T0':
                    label_list_list.append(label_list[i])
                    seg_start_list.append(seg_start[i])
            label_list = np.array(label_list_list)
            seg_start = np.array(seg_start_list)
            seg_length = len(label_list)
            for seg in range(seg_length):
                start = int(seg_start[seg])
                seg_buff_all[sample_count, :, :] = numpy_data[:, start: (start + 640)]
                label_buff_all[sample_count] = int(label_list[seg][-1]) - 1
                sample_count += 1
    #print(sample_count)
    # new_buff = [seg_buff_all[i * 30:i * 30 + 24, :, :] for i in range(50)]
    # buff_temp = np.vstack((new_buff[0], new_buff[1]))
    # cont_range = 50
    # for cont in range(2, cont_range):
    #     buff_temp = np.vstack((buff_temp, new_buff[cont]))
    # 
    # new_label = [label_buff_all[i * 30:i * 30 + 24] for i in range(50)]
    # buff_label = np.hstack((new_label[0], new_label[1]))
    # for cont in range(2, cont_range):
    #     buff_label = np.hstack((buff_label, new_label[cont]))
    # 
    # new_buff_test = [seg_buff_all[i * 30 + 24:(i + 1) * 30, :, :] for i in range(50)]
    # buff_temp_test = np.vstack((new_buff_test[0], new_buff_test[1]))
    # for cont in range(2, cont_range):
    #     buff_temp_test = np.vstack((buff_temp_test, new_buff_test[cont]))
    # 
    # new_label_test = [label_buff_all[i * 30 + 24:(i + 1) * 30] for i in range(50)]
    # buff_label_test = np.hstack((new_label_test[0], new_label_test[1]))
    # for cont in range(2, cont_range):
    #     buff_label_test = np.hstack((buff_label_test, new_label_test[cont]))
    #return buff_temp, buff_label, buff_temp_test, buff_label_test
    return seg_buff_all, label_buff_all, sample_count
