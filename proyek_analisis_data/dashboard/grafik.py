import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

def grafik_distribusi(data) :
    
    continuous_columns = ['temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered', 'cnt']

    fig, ax = plt.subplots(2, 4, figsize = (10, 8.4))
    
    fig.suptitle('Distribusi Nilai')

    baris, kolom = 0, 0

    # Membuat dataframe kosong untuk tempat menyimpan nilai skew
    shape_data = pd.DataFrame() 

    # Menampilkan histogram dari semua kolom continuous
    for idx, cont in enumerate(continuous_columns) :

        if idx % 2 == 0 :
            color = warna_default = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]
        else :
            color = 'orange'

        # Memasukkan nilai skew dan kurt ke dataframe yang akan ditampilkan di halaman kosong
        skewn = data[cont].skew()
        kurto = data[cont].kurt()

        shape_data.loc[cont, 'skew'] = skewn
        shape_data.loc[cont, 'kurt'] = kurto

        ax[baris, kolom].hist(data[cont], edgecolor = 'black', color = color)
        ax[baris, kolom].set_title(cont)

        if kolom < 3 :
            kolom += 1
        else :
            baris +=1
            kolom = 0

    ax[1, 3].axis('off') # Menghilangkan sumbu

    # Membuat tabel yang berisi nilai-nilai di shape_data
    table = ax[1, 3].table(cellText = [[col, round(shape_score[0], 2), round(shape_score[1], 2)] for col, shape_score in zip(shape_data.index, shape_data.values)], colLabels = ['', 'skew', 'kurt'], cellLoc = 'center', loc = 'center')

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.4, 2.2) 

    plt.tight_layout()
    
    
    return fig

def pertanyaan_1(data) :
    fig, ax = plt.subplots(2, 2, figsize = (10, 7.5)) 

    col_to_vis = ['yr', 'mnth', 'weekday', 'date']
    title      = ['Tahun', 'Bulan', 'Hari', 'Tanggal']

    baris, kolom  = 0, 0
    warna_default = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]

    for i, col in enumerate(col_to_vis) :

        if col == 'yr':
            data.groupby(col)['cnt'].mean().plot(kind = 'bar', ax = ax[baris, kolom], color = [warna_default, 'orange'])
        else :

            for tahun, color in zip(data.yr.unique(), [warna_default, 'orange']) :

                per_tahun = data[data.yr == tahun].groupby(col)['cnt'].mean()
                per_tahun.plot(color = color, kind = 'line', linestyle = 'dotted', marker = 'o', ax = ax[baris, kolom], label = tahun)

        ax[baris, kolom].set_ylim(0, 8000)
        ax[baris, kolom].set_title('Tren Total Penyewaan berdasarkan {}'.format(title[i]))
        ax[baris, kolom].axhline(y = data.cnt.mean(), color = 'red', linestyle = '--') # Garis rata-rata keseluruhan pelanggan
        ax[baris, kolom].grid(axis = 'x')

        if kolom < 1 :
            kolom += 1
        else :
            baris += 1
            kolom = 0

    ax[1, 0].legend(loc = 'upper left')
    plt.tight_layout()
    
    return fig

def pertanyaan_2(data) :
    
    fig = plt.figure(figsize = (10, 7.5))
    gs  = fig.add_gridspec(2, 3)
    
    warna_default = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]

    baris, kolom = 0, 0
    color = 'orange'
    for col in ['temp', 'atemp', 'windspeed', 'hum'] :
        ax = fig.add_subplot(gs[baris, kolom])

        sns.regplot(x = data[col], y = data['cnt'], ax = ax, color = color)

        if kolom < 1 :
            kolom += 1
        else :
            baris += 1
            kolom = 0
            color = warna_default

    ax_ = fig.add_subplot(gs[:,2])

    cmap = LinearSegmentedColormap.from_list('custom_cmap', [warna_default, 'white', 'orange'])

    corr_df = pd.DataFrame(data.corr(numeric_only = True).loc[['temp', 'atemp', 'windspeed', 'hum'], 'cnt'])

    sns.heatmap(corr_df, ax = ax_, annot = True, square = True, fmt = ".3f", cmap = cmap, cbar = True, label = True)

    plt.tight_layout()
    
    return fig

def pertanyaan_3(data) :
    fig, ax = plt.subplots(2, 2, figsize = (10, 5)) 

    baris, kolom = 0, 0

    for s in np.unique(data['season']) :

        count = data[data['season'] == s].groupby('hr')['cnt'].mean()

        # Mengambil index dengan nilai 3 terbesar
        top_three = count.nlargest(3)

        warna_default = plt.rcParams['axes.prop_cycle'].by_key()['color'][0] # --> Digunakan untuk mengambil warna default pertama dari siklus warna matplotlib

        # warna top three --> orange, selain itu : warna default matplotlib (biru)
        colors = ['orange' if idx in top_three else warna_default for idx in count.index]

        ax[baris, kolom].bar(range(0, 24), count, color = colors)
        ax[baris, kolom].set_title(f'Musim {s}')

        if baris == 1 :
            ax[baris, kolom].set_xlabel('Jam')

        if kolom == 0 :
            ax[baris, kolom].set_ylabel('Count')

        ax[baris, kolom].set_ylim(0, 600)

        if kolom < 1 :
            kolom += 1
        else :
            baris += 1
            kolom = 0

    plt.tight_layout()
    
    return fig

def pertanyaan_4(data, data_) :
    fig = plt.figure(figsize = (10, 10))
    gs  = fig.add_gridspec(4, 2)

    baris, kolom = 0, 0

    hari = np.unique(data['weekday'])
    
    for day in hari :
        ax = fig.add_subplot(gs[baris, kolom])

        count = data[data['weekday'] == day].groupby('hr')['cnt'].mean()

        # Mengambil index dengan nilai 3 terbesar
        top_three = count.nlargest(3)

        warna_default = plt.rcParams['axes.prop_cycle'].by_key()['color'][0] # --> Digunakan untuk mengambil warna default pertama dari siklus warna matplotlib

        colors = ['orange' if idx in top_three else warna_default for idx in count.index]

        ax.bar(range(0, 24), count, color = colors)
        ax.set_title(f'{day}')

        ax.set_ylim(0, 600)

        if kolom < 1 :
            kolom += 1
        else :
            baris += 1
            kolom = 0

    rerata_tiap_hari = data_.groupby('weekday')['cnt'].mean()
    top_three = rerata_tiap_hari.nlargest(3)

    colors = ['orange' if idx in top_three else warna_default for idx in rerata_tiap_hari.index]

    ax_ = fig.add_subplot(gs[3, 1])
    ax_.barh(width = rerata_tiap_hari.values, y = rerata_tiap_hari.index, color = colors)

    plt.tight_layout()
    
    return fig
