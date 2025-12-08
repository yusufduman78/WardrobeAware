import json
from pathlib import Path


def update_notebook(path: Path) -> None:
    nb = json.loads(path.read_text(encoding='utf-8'))

    # Replace category distribution cell
    if len(nb['cells']) > 5 and nb['cells'][5].get('cell_type') == 'code':
        nb['cells'][5]['source'] = [
            "plt.figure(figsize=(18, 8))\n",
            "sns.countplot(y='category_name', data=df, order=df['category_name'].value_counts().index, hue='category_name', palette='viridis', legend=False)\n",
            "plt.title('Distribution of Clothing Categories', fontsize=16, weight='bold')\n",
            "plt.xlabel('Number of Items', fontsize=12)\n",
            "plt.ylabel('Category', fontsize=12)\n",
            "plt.tight_layout()\n",
            "plt.show()\n"
        ]

    # Ensure subplot creation cell is consistent
    if len(nb['cells']) > 7 and nb['cells'][7].get('cell_type') == 'code':
        nb['cells'][7]['source'] = [
            "fig, axes = plt.subplots(1, 3, figsize=(20, 6))\n",
            "fig.suptitle('Distribution of Image Attributes', fontsize=18, weight='bold')\n"
        ]

    # Rewrite attribute distribution cell
    if len(nb['cells']) > 8 and nb['cells'][8].get('cell_type') == 'code':
        nb['cells'][8]['source'] = [
            "# Distribution of annotation attributes\n",
            "scale_map = {1: 'Small', 2: 'Modest', 3: 'Large'}\n",
            "df['scale_label'] = df['scale'].map(scale_map)\n",
            "occlusion_map = {1: 'Slight/None', 2: 'Medium', 3: 'Heavy'}\n",
            "df['occlusion_label'] = df['occlusion'].map(occlusion_map)\n",
            "viewpoint_map = {1: 'No Wear', 2: 'Frontal', 3: 'Side/Back'}\n",
            "df['viewpoint_label'] = df['viewpoint'].map(viewpoint_map)\n",
            "sns.countplot(ax=axes[0], x='scale_label', data=df, order=['Small', 'Modest', 'Large'], hue='scale_label', palette='magma', legend=False)\n",
            "axes[0].set_title('Item Scale')\n",
            "axes[0].set_xlabel('')\n",
            "sns.countplot(ax=axes[1], x='occlusion_label', data=df, order=['Slight/None', 'Medium', 'Heavy'], hue='occlusion_label', palette='magma', legend=False)\n",
            "axes[1].set_title('Item Occlusion')\n",
            "axes[1].set_xlabel('')\n",
            "sns.countplot(ax=axes[2], x='viewpoint_label', data=df, order=['No Wear', 'Frontal', 'Side/Back'], hue='viewpoint_label', palette='magma', legend=False)\n",
            "axes[2].set_title('Item Viewpoint')\n",
            "axes[2].set_xlabel('')\n",
            "plt.tight_layout(rect=[0, 0, 1, 0.96])\n",
            "plt.show()\n"
        ]

    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
    print('Notebook updated:', path)


if __name__ == '__main__':
    update_notebook(Path('../notebooks/deepfashion-eda.ipynb'))
