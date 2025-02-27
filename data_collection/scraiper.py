import json
import os

from data_collection.scrape_utils import Clowl


def main(usr_id):
    file_path = "data/all_users_results.json"
    if os.path.isfile(file_path):
        with open(file_path, "r") as file:
            data = json.load(file)
        user_id = list(data.keys())[-1]
        print(f"{user_id}からクローリングを再開します")
    else:
        user_id = "00"
    clowler = Clowl(start_usr_id=usr_id, num_usrs=1000, data=data)
    clowler.excute()


if __name__ == "__main__":
    main(usr_id="")
