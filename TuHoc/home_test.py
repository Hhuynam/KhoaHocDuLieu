import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
dulieu={"Ho Ten":["Ha Huy","Nam Ha","Pham Nhung"],
        "Tuoi":[23,15,16],
        "Que Quan":["Thai Nguyen","Bac Giang","Quang Ninh"]}
df=pd.DataFrame(dulieu)
#Ve bieu do cot
plt.bar(df["HoTen"],df["Tuoi"],color="skyblue")
#Them nhan du lieu
plt.xlabel("HoTen")
plt.ylabel("Tuoi")
plt.title("Bieu do abc")
plt.show()