import json
import re  # noqa
import time  # noqa
from urllib.error import HTTPError, URLError
from urllib.request import urlopen  # noqa

import requests
from bs4 import BeautifulSoup
from selenium.common.exceptions import StaleElementReferenceException  # noqa
from selenium.webdriver.chrome.service import Service  # noqa
from tqdm import tqdm


class Clowl:
    def __init__(self, start_usr_id: str, num_usrs: int = 10, data: dict = {}):
        self.start_usr_id = start_usr_id
        self.num_usrs = num_usrs
        self.all_users_result = {}
        self.data = data

    def excute(self):
        try:
            target_user = self.start_usr_id
            total = 0
            print("__読書履歴のクローリングを開始します__")
            while total <= self.num_usrs:
                target_user = int(target_user) + 1
                gender = self.gender_label(target_user)
                time.sleep(10)
                success_Flag, user_books = self.clowler(user_id=str(target_user), gender=gender)

                if success_Flag:
                    self.all_users_result[str(target_user)] = user_books
                    total += 1
                    continue
        except KeyboardInterrupt:
            print("\nクローリングを中断します。途中結果を保存します。")
            self.save_json()  # ここで途中までのデータを保存
            print("途中結果を保存しました。")
        except requests.exceptions.RequestException as e:
            print(f"\nネットワークエラー: {e}")
            print("途中結果を保存します。")
            self.save_json()
            print("途中結果を保存しました。")
        except Exception:
            print("\nクローリングが中断されました。途中結果を保存します。")
            self.save_json()  # ここで途中までのデータを保存
            print("途中結果を保存しました。")
        self.save_json()

    def clowler(self, user_id: str, gender: str, max_retries: int = 3, retry_delay: int = 5):
        user_books = []
        base_url = f"https://bookmeter.com/users/{user_id}/books/"

        retries = 0
        while retries < max_retries:
            try:
                time.sleep(10)
                read_url = base_url + "read"
                request = urlopen(read_url)
                html = request.read()
                break  # Exit the loop if the request is successful
            except HTTPError as e:
                if e.code == 503:
                    print(f"Service Unavailable in {user_id}. Retrying in {retry_delay} seconds...")
                    retries += 1
                    time.sleep(retry_delay)
                else:
                    print(f"{user_id}でHTTPエラーが以下の理由で発生しました{e.reason}")
                    return False, user_books
            except URLError as e:
                print(f"URLエラーが以下の理由で発生しました{e.reason}")
            except Exception as e:
                print(f"なにかしらのエラーが発生しました:{e}")
        if retries == max_retries:
            print("最大リトライ回数に達しました。終了します。")
            return False, user_books
        # 最終ページにアクセス
        length = self.last_page(html, user_id)

        # 読書履歴の捜索
        for i in tqdm(
            range(1, length + 1), desc="Page Transition", leave=False
        ):  # ここを修正(最初ページの本を2回そのままとってきてた)
            url = f"https://bookmeter.com/users/{user_id}/books/read?page={i}"
            time.sleep(30)
            request = urlopen(url)
            html = request.read()

            bs = BeautifulSoup(html, "html.parser")

            books = bs.find_all("li", class_="group__book")

            links_list = [
                a
                for a in bs.find("div", class_="content-with-header__content").find_all(
                    "a", href=re.compile("^(/books/)((?!:).)*$")
                )
                if not a.find("img")
            ]
            for book_item, book_id in tqdm(
                zip(books, links_list), desc="Processing Books", leave=False
            ):
                # タイトルと著者の取得
                img_tag = book_item.find("img", class_="cover__image")
                title = (
                    img_tag.get("alt")
                    if img_tag
                    else book_item.find("div", class_="detail__title").get_text(strip=True)
                )
                author = book_item.find("ul", class_="detail__authors").get_text(strip=True)
                book_data = {"Title": title, "Author": author, "Gender": gender}
                user_books.append(book_data)
        return True, user_books

    def last_page(self, html, user_id):
        bs = BeautifulSoup(html, "html.parser")
        last_page_link = bs.find("a", text="最後")
        if last_page_link is not None:
            href = last_page_link.attrs["href"]
            match = re.search(r"page=(\d+)", href)
            if match:
                page_number = match.group(1)
                print(f"{user_id}のページ数は{page_number}です")
                return int(page_number)

            else:
                raise ValueError("The page number could not be found in the href.")
        else:
            print(f"{user_id}のページ数は{1}です")
            return 1

    def gender_label(self, target_user):
        usr_url = f"https://bookmeter.com/users/{target_user}"
        try:
            time.sleep(5)
            request = urlopen(usr_url)
            html = request.read()

            soup = BeautifulSoup(html, "html.parser")
            gender_label = soup.find("dt", class_="bm-details-side__title", string="性別")
            gender_value = gender_label.find_next("dd", class_="bm-details-side__item").get_text()

            if gender_value == "女":
                return "F"
            else:
                return "M"
        except Exception:
            return "N"

    def save_json(self):
        if self.data and self.all_users_result:
            self.all_users_result = self.data.update(self.all_users_result)
        """全ユーザーのクローリング結果を JSON ファイルに保存する"""
        with open("data/all_users_results.json", "w", encoding="utf-8") as f:
            json.dump(self.all_users_result, f, ensure_ascii=False, indent=4)
        print("結果を all_users_results.json として保存しました。")
