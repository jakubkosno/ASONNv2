from fileinput import close, filename


class DatasetLoader:
    def load_from_file(self, file_path):
        db_content = []
        try:
            with open(file_path) as f:
                lines = f.readlines()
            f.close()
        except:
            return db_content
        for line in lines:
            db_content.append(self.parse_db_record(line))
        
        db_content[0].insert(0, "Class")
        return db_content

    def parse_db_record(self, record_str):
        record = list()
        for value in record_str.split(","):
            try:
                record.append(float(value))
            except:
                record.append(value)

        return record
