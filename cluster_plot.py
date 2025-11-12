# given photos, detect and embed faces, then cluster them
# generate the plot of the clustering
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from utils import *

def plot_cluster(embeddings, clustering):
    # reduce embedding dimensions to 2D for plotting
    reducer = TSNE(n_components=2, perplexity=5)
    points_2d = reducer.fit_transform(np.array(embeddings))
    cluster_num = max(clustering)+1

    # plot
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        points_2d[:, 0],
        points_2d[:, 1],
        c=clustering,
        cmap=plt.get_cmap('tab10', cluster_num),
        norm=mcolors.BoundaryNorm(np.arange(0, cluster_num + 1) - 0.5, cluster_num),
        s=30
    )
    tick_locs = np.arange(0, cluster_num)
    cbar = plt.colorbar(scatter, ticks=tick_locs, label="Cluster ID")
    cbar.formatter = mticker.FormatStrFormatter('%d')
    cbar.update_ticks()
    plt.title("face embedding cluster")
    plt.show()


embeddings = []

## replace "photos/photo1.jpg" with path to any photo
faces = []
img_list = [
    "../photos/image.png", 
    "../photos/left.png", 
    "../photos/photo1.jpg",
    "../photos/right.png", 
    "../photos/trump_test.jpg", 
    "../photos/trump_test2.jpg", 
    "../photos/trump_test3.jpg", 
    "../photos/trump.jpg", 
]

for img_path in img_list:
    faces += detect_and_embed(img_path)

for f in faces:
    embeddings.append(f["embedding"])
clustering = cluster(embeddings)
print(clustering)
# print(clustering)
plot_cluster(embeddings, clustering)