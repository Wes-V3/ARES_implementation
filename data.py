import functools
import logging

import numpy as np
import pandas as pd
import torch

import atom3d.datasets as da
import atom3d.util.rosetta as rose
import e3nn.point.data_helpers as dh


logger = logging.getLogger("lightning")


def create_transform(use_labels, label_dir, filetype):
    if use_labels and not label_dir and filetype == 'pdb':
        raise RuntimeError('No label path specified for filetype pdb.')

    label_to_use = 'rms' if use_labels else None
    transform = functools.partial(prepare, label_to_use=label_to_use)

    if label_dir:
        label_paths = da.get_file_list(label_dir, 'sc')
        logger.info(f'Deriving labels from {label_dir} and found '
                    f'{len(label_paths)} score files...')
        labels = rose.Scores(label_paths)
        return lambda x: transform(labels(x))
    elif use_labels:
        logger.info('Assuming labels are already in dataset...')
        return transform
    else:
        return transform


def prepare(item, k=50, label_to_use='rms'):
    element_mapping = {
        'C': 0,
        'O': 1,
        'N': 2,
    }
    num_channels = len(element_mapping)
    if type(item['atoms']) != pd.DataFrame:
        item['atoms'] = pd.DataFrame(**item['atoms'])
    coords = item['atoms'][['x', 'y', 'z']].values
    elements = item['atoms']['element'].values

    if label_to_use is None:
        # Don't use any label.
        label = [0]
    else:
        #以下两句我自己加的，注释掉的是原本的程序。这里的程序需要配合我写的add_labels_from_farfar2pdbs.py。
        scores = item['scores']
        label = [scores]

        # scores = item['scores']
        # if type(scores) != pd.Series and 'data' in scores \
        #         and 'index' in scores:
        #     scores = pd.Series(
        #         scores['data'], index=scores['index'], name=item['id'])
        # else:
        #     scores = pd.Series(scores, index=scores.keys(), name=item['id'])
        # label = [scores[label_to_use]]

    sel = np.array([i for i, e in enumerate(elements) if e in element_mapping])
    total_atoms = elements.shape[0]
    coords = coords[sel]
    elements = elements[sel]

    # Make one-hot
    elements_int = np.array([element_mapping[e] for e in elements])
    one_hot = np.zeros((elements.size, len(element_mapping)))
    one_hot[np.arange(elements.size), elements_int] = 1

    geometry = torch.tensor(coords, dtype=torch.float32)
    features = torch.tensor(one_hot, dtype=torch.float32)
    label = torch.tensor(label)

    ra = geometry.unsqueeze(0)
    rb = geometry.unsqueeze(1)
    pdist = (ra - rb).norm(dim=2)
    tmp = torch.topk(-pdist, k, axis=1)

    nei_list = []
    geo_list = []
    for source, x in enumerate(tmp.indices):
        cart = geometry[x]
        nei_list.append(
            torch.tensor(
                [[source, dest] for dest in x], dtype=torch.long))
        geo_list.append(cart - geometry[source])
    nei_list = torch.cat(nei_list, dim=0).transpose(1, 0)
    geo_list = torch.cat(geo_list, dim=0)

    r_max = 10  # Doesn't matter since we override
    d = dh.DataNeighbors(features, [(num_channels, 0)], geometry, r_max)
    d.edge_attr = geo_list
    d.edge_index = nei_list
    d.label = label
    d.id = item['id']
    d.file_path = item['file_path']

    return d
