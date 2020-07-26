import csv
from collections import defaultdict


class LabelDataTable:
    def load_csv_data(self, file_path, direction):
        data_table = None
        if direction == 'c':
            data_table = defaultdict(list)
            with open('%s' % file_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    for (k, v) in row.items():
                        data_table[k].append(v)
                f.close()
        elif direction == 'r':
            index = -1
            data_table = defaultdict(list)
            with open('%s' % file_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    index = index + 1
                    if index == 0:
                        for (k, v) in row.items():
                            data_table['name'].append(k)
                            data_table[str(index)].append(v)
                    else:
                        for (k, v) in row.items():
                            data_table[str(index)].append(v)
                f.close()
        return data_table

    def get_dict_keys(self, defaultdict_table):
        return [i for i in defaultdict_table if defaultdict_table[i] != defaultdict_table.default_factory()]
