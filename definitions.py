import os
import seaborn as sns


ROOT_FOLDER = os.path.dirname(os.path.abspath(__file__))
RESULTS_FOLDER = os.path.join(ROOT_FOLDER, 'results')
COLOR_PALETTE = sns.color_palette('Set1')