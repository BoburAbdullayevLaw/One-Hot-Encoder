from sklearn.preprocessing import OneHotEncoder
import numpy as np


def one_hot_soz(sozlar: list[str]):
    matnlar = np.array(sozlar).reshape(-1, 1)

    # sklearn yangi versiyasida sparse 
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder.fit(matnlar)

    print(" ONE-HOT NATIJASI:")
    print(encoder.transform(matnlar))
    print("\n📌 So'zlar tartibi:")
    print(encoder.get_feature_names_out())

    def kodla(yangi_soz: str):
        return encoder.transform([[yangi_soz]])[0]

    return kodla


if __name__ == "__main__":
    print("TAYYORLANMOQDA ONE-HOT ENCODING")
    corpus = ["salom", "dunyo", "salom", "qanday", "salom", "yaxshi"]
    kodlash = one_hot_soz(corpus)

    print(f"\n'salom' → {kodlash('salom')}")
    print(f"'yaxshi' → {kodlash('yaxshi')}")
    print(f"'nima' → {kodlash('nima')} (noma'lum → 0)")

    print("\n TUGADI!")
