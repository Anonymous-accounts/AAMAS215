import os

def read_tfevent_lists(path, all_files):
    file_list = os.listdir(path)
    for file in file_list:
        cur_path = os.path.join(path, file)
        if os.path.isdir(cur_path):
            read_tfevent_lists(cur_path, all_files)
        else:
            all_files.append(path + "/" + file)
    return all_files

def return_npy_events_list(path):
    tfevents = read_tfevent_lists(path, [])
    npy_events_list = []
    for event in tfevents:
        if '.npy' in event:
            npy_events_list.append(event)
    return npy_events_list


if __name__ == "__main__":
    data_list = []
    path_root = '/home/username/data/BABYAI/best_data'

    ssg1_data = return_npy_events_list(path_root)
    print(ssg1_data)



