import json
import os

from data_collection.scrape_utils import Clowl


def run_scrape(usr_id=None):
    file_path = "data/all_users_results.json"
    if os.path.isfile(file_path):
        with open(file_path, "r") as file:
            data = json.load(file)
        if len(data) >= 1:
            # ✅ 辞書のキーは文字列なので、数値として比較するためにint変換
            user_id = str(max(map(int, data.keys())))
        else:
            user_id = "00"
    else:
        data = {}
        user_id = "00"

    # ✅ usr_idがNoneや空文字だった場合、自動的に user_id を使う
    if not usr_id:
        usr_id = user_id

    print(f"{usr_id}からクローリングを再開します")
    clowler = Clowl(start_usr_id=usr_id, num_usrs=1000, data=data)
    clowler.excute()


if __name__ == "__main__":
    run_scrape()
