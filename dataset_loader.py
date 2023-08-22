from fileinput import close, filename


class DatasetLoader:
    def load_from_file(self, file_path):
        dataset = []
        try:
            with open(file_path) as f:
                lines = f.readlines()
            f.close()
        except:
            return dataset
        for line in lines:
            dataset.append(self.parse_db_record(line))
        
        dataset[0].insert(0, "Class")
        return dataset

    def parse_db_record(self, record_str):
        record = list()
        for value in record_str.split(","):
            try:
                record.append(float(value))
            except:
                record.append(value)

        return record
