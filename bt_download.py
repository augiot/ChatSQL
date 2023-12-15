import libtorrent as lt
import time

def download_magnet(magnet_link, save_path):
    ses = lt.session()
    info = lt.parse_magnet_uri(magnet_link)
    h = ses.add_torrent({"ti": info, "save_path": save_path})
    print("下载中...")

    while not h.is_seed():
        s = h.status()
        print("进度： %.2f%%" % (s.progress * 100))
        time.sleep(1)

    print("下载完成！")

if __name__ == "__main__":
    # magnet_link = input("请输入磁力链接：")
    # save_path = input("请输入保存路径：")
    magnet_link = "magnet:?xt=urn:btih:b5fed4ee16b1c9b886f44dd30cd70fdf746f58f5&dn=zh-cn_windows_11_business_editions_version_23h2_x64_dvd_2a79e0f1.iso&xl=6613571584"
    save_path = "./"
    download_magnet(magnet_link, save_path)
