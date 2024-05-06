import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

def add_subtitle(ax:plt.axes, subtitle:str, location='upper left', alpha=1.0) -> plt.axes:
    anchored_text = AnchoredText(subtitle, loc=location, borderpad=0.0)
    anchored_text.patch.set_alpha(alpha)
    anchored_text.zorder = 15
    ax.add_artist(anchored_text)
    return ax