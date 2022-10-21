import glob


def get_sample_list():
    path = 'examples/*.csv'
    files = glob.glob(path)
    files = [x.replace('.csv', '') for x in files]
    print(files)


def process_sample_1st_line(lst_name):
    for item in lst_name:
        jpg_name, csv_name = item + '.jpg', item + '.csv'
